#include "tamev/tamev_classifier.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "core/paths.hpp"
#include "tamev/bert_wordpiece_tokenizer.hpp"
#include "tamev/option_catalog.hpp"

#ifdef MIRA_HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace mira
{

struct TamevClassifier::Impl
{
    BertWordPieceTokenizer tokenizer;
    std::vector<EncodedTokens> encoded_options;

#ifdef MIRA_HAVE_ONNXRUNTIME
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::string ctx_ids_name = "ctx_input_ids";
    std::string ctx_mask_name = "ctx_attention_mask";
    std::string opt_ids_name = "opt_input_ids";
    std::string opt_mask_name = "opt_attention_mask";
#endif

    bool loaded = false;
    std::string load_error;
};

namespace
{

ClassificationResult unknown_result(std::size_t option_count)
{
    ClassificationResult out;
    out.intent = Intent::Unknown;
    out.confidence = 0.0f;
    out.probabilities.assign(option_count, 0.0f);
    return out;
}

ClassificationResult from_probabilities(const std::vector<float>& probs, float threshold)
{
    ClassificationResult out;
    out.probabilities = probs;
    if (probs.empty())
    {
        out.intent = Intent::Unknown;
        out.confidence = 0.0f;
        return out;
    }

    std::size_t best = 0;
    for (std::size_t i = 1; i < probs.size(); ++i)
    {
        if (probs[i] > probs[best])
        {
            best = i;
        }
    }
    out.confidence = probs[best];
    out.intent = (out.confidence >= threshold) ? tamev_index_to_intent(best)
                                               : Intent::Unknown;
    return out;
}

#ifdef MIRA_HAVE_ONNXRUNTIME
std::vector<float> softmax_last_k(const float* data, std::size_t k)
{
    std::vector<float> probs(k, 0.0f);
    if (k == 0 || data == nullptr)
    {
        return probs;
    }
    float max_logit = data[0];
    for (std::size_t i = 1; i < k; ++i)
    {
        max_logit = std::max(max_logit, data[i]);
    }
    float sum = 0.0f;
    for (std::size_t i = 0; i < k; ++i)
    {
        probs[i] = std::exp(data[i] - max_logit);
        sum += probs[i];
    }
    if (sum > 0.0f)
    {
        for (float& p : probs)
        {
            p /= sum;
        }
    }
    return probs;
}
#endif

} // namespace

TamevClassifier::TamevClassifier(float confidence_threshold)
    : impl_(new Impl()), confidence_threshold_(confidence_threshold)
{
}

TamevClassifier::~TamevClassifier() = default;

Result TamevClassifier::load_model(const std::string& onnx_model_path,
                                   const std::string& tokenizer_json_path)
{
    impl_ = std::unique_ptr<Impl>(new Impl());

    Result tok_result = impl_->tokenizer.load(tokenizer_json_path);
    if (tok_result.is_error())
    {
        impl_->load_error = tok_result.to_string();
        return Result::unavailable("tamev tokenizer unavailable", impl_->load_error);
    }

    if (!paths::file_exists(onnx_model_path))
    {
        impl_->load_error = "missing onnx file: " + onnx_model_path;
        return Result::unavailable("tamev model unavailable", impl_->load_error);
    }

    // Pre-encode the fixed option list once (options never change per query).
    const std::vector<std::string>& options = tamev_option_strings();
    impl_->encoded_options.clear();
    impl_->encoded_options.reserve(options.size());
    for (const std::string& option : options)
    {
        impl_->encoded_options.push_back(
            impl_->tokenizer.encode(option, TAMEV_OPTION_LENGTH));
    }

#ifdef MIRA_HAVE_ONNXRUNTIME
    try
    {
        impl_->env = std::unique_ptr<Ort::Env>(
            new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "mira-tamev"));
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
        impl_->session = std::unique_ptr<Ort::Session>(new Ort::Session(
            *impl_->env, onnx_model_path.c_str(), session_options));

        // Confirm the inference contract at load time instead of assuming it.
        Ort::AllocatorWithDefaultOptions allocator;
        if (impl_->session->GetInputCount() < 4 || impl_->session->GetOutputCount() < 1)
        {
            impl_->load_error = "unexpected tamev graph (inputs/outputs)";
            impl_->session.reset();
            impl_->env.reset();
            return Result::unavailable("tamev model unavailable", impl_->load_error);
        }
        auto read_input_name = [&](size_t i, const char* fallback) {
            try
            {
                Ort::AllocatedStringPtr s =
                    impl_->session->GetInputNameAllocated(i, allocator);
                return s.get() != nullptr ? std::string(s.get()) : std::string(fallback);
            }
            catch (...)
            {
                return std::string(fallback);
            }
        };
        impl_->ctx_ids_name = read_input_name(0, "ctx_input_ids");
        impl_->ctx_mask_name = read_input_name(1, "ctx_attention_mask");
        impl_->opt_ids_name = read_input_name(2, "opt_input_ids");
        impl_->opt_mask_name = read_input_name(3, "opt_attention_mask");
    }
    catch (const Ort::Exception& e)
    {
        impl_->load_error = e.what();
        impl_->session.reset();
        impl_->env.reset();
        return Result::unavailable("tamev model unavailable", impl_->load_error);
    }
    catch (const std::exception& e)
    {
        impl_->load_error = e.what();
        return Result::unavailable("tamev model unavailable", impl_->load_error);
    }
#else
    impl_->load_error = "built without onnxruntime support";
    return Result::unavailable("tamev model unavailable", impl_->load_error);
#endif

    impl_->loaded = true;
    return Result::ok();
}


ClassificationResult TamevClassifier::classify(const std::string& utterance) const
{
    const std::size_t option_count = tamev_option_strings().size();
    if (impl_ == nullptr || !impl_->loaded)
    {
        return unknown_result(option_count);
    }

#ifdef MIRA_HAVE_ONNXRUNTIME
    try
    {
        if (impl_->session == nullptr)
        {
            return unknown_result(option_count);
        }

        const std::size_t k = impl_->encoded_options.size();
        if (k == 0)
        {
            return unknown_result(option_count);
        }

        EncodedTokens ctx = impl_->tokenizer.encode(
            tamev_context_text(utterance), TAMEV_CONTEXT_LENGTH);

        std::vector<int64_t> opt_ids;
        std::vector<int64_t> opt_mask;
        opt_ids.reserve(k * TAMEV_OPTION_LENGTH);
        opt_mask.reserve(k * TAMEV_OPTION_LENGTH);
        for (const EncodedTokens& enc : impl_->encoded_options)
        {
            opt_ids.insert(opt_ids.end(), enc.input_ids.begin(), enc.input_ids.end());
            opt_mask.insert(opt_mask.end(), enc.attention_mask.begin(), enc.attention_mask.end());
        }

        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<int64_t> ctx_shape{1, static_cast<int64_t>(TAMEV_CONTEXT_LENGTH)};
        std::vector<int64_t> opt_shape{
            1, static_cast<int64_t>(k), static_cast<int64_t>(TAMEV_OPTION_LENGTH)};

        Ort::Value ctx_ids = Ort::Value::CreateTensor<int64_t>(
            memory_info, ctx.input_ids.data(), ctx.input_ids.size(),
            ctx_shape.data(), ctx_shape.size());
        Ort::Value ctx_mask = Ort::Value::CreateTensor<int64_t>(
            memory_info, ctx.attention_mask.data(), ctx.attention_mask.size(),
            ctx_shape.data(), ctx_shape.size());
        Ort::Value opt_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, opt_ids.data(), opt_ids.size(),
            opt_shape.data(), opt_shape.size());
        Ort::Value opt_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, opt_mask.data(), opt_mask.size(),
            opt_shape.data(), opt_shape.size());

        const char* input_names[4] = {
            impl_->ctx_ids_name.c_str(), impl_->ctx_mask_name.c_str(),
            impl_->opt_ids_name.c_str(), impl_->opt_mask_name.c_str()};
        Ort::Value inputs[4] = {std::move(ctx_ids), std::move(ctx_mask),
                                std::move(opt_ids_tensor), std::move(opt_mask_tensor)};

        // Prefer the calibrated "probs" output, fall back to softmax(logits).
        Ort::AllocatorWithDefaultOptions allocator;
        const size_t out_count = impl_->session->GetOutputCount();
        std::vector<std::string> out_storage;
        out_storage.reserve(out_count);
        for (size_t i = 0; i < out_count; ++i)
        {
            Ort::AllocatedStringPtr s =
                impl_->session->GetOutputNameAllocated(i, allocator);
            out_storage.push_back(s.get() != nullptr ? s.get() : "logits");
        }
        std::vector<const char*> output_names;
        output_names.reserve(out_storage.size());
        for (const std::string& name : out_storage)
        {
            output_names.push_back(name.c_str());
        }

        std::vector<Ort::Value> outputs = impl_->session->Run(
            Ort::RunOptions{nullptr}, input_names, inputs, 4,
            output_names.data(), output_names.size());

        std::vector<float> probs;
        for (size_t i = 0; i < out_storage.size() && i < outputs.size(); ++i)
        {
            Ort::TensorTypeAndShapeInfo info = outputs[i].GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                continue;
            }
            const float* data = outputs[i].GetTensorData<float>();
            if (data == nullptr)
            {
                continue;
            }
            size_t count = 1;
            for (int64_t d : info.GetShape())
            {
                count *= static_cast<size_t>(d > 0 ? d : 1);
            }
            if (count < k)
            {
                continue;
            }
            const float* row = data + (count - k); // batch size 1: last row
            if (out_storage[i] == "probs")
            {
                probs.assign(row, row + k);
                break;
            }
            if (probs.empty())
            {
                probs = softmax_last_k(row, k);
            }
        }
        if (probs.size() != k)
        {
            return unknown_result(option_count);
        }
        return from_probabilities(probs, confidence_threshold_);
    }
    catch (...)
    {
        return unknown_result(option_count);
    }
#else
    return unknown_result(option_count);
#endif
}

bool TamevClassifier::is_loaded() const
{
    return impl_ != nullptr && impl_->loaded;
}

float TamevClassifier::confidence_threshold() const
{
    return confidence_threshold_;
}

void TamevClassifier::set_confidence_threshold(float threshold)
{
    confidence_threshold_ = threshold;
}

} // namespace mira
