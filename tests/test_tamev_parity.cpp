// Parity test: C++ classifier vs frozen Python reference vectors.
// models/parity/frozen_vectors.json pins the full path for the fixed
// ONNX artifact. Requires MIRA_HAVE_ONNXRUNTIME; otherwise skipped.

#include "tamev/tamev_classifier.hpp"
#include "tamev/option_catalog.hpp"
#include "core/paths.hpp"
#include "test_util.hpp"

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace
{

void parity_against_frozen_vectors()
{
#ifndef MIRA_HAVE_ONNXRUNTIME
    std::cout << "tamev_parity: built without onnxruntime, skipping\n";
    return;
#else
    using json = nlohmann::json;
    const std::string path =
        mira::paths::resolve_data_path("models/parity/frozen_vectors.json");
    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cout << "tamev_parity: frozen vectors missing, skipping\n";
        return;
    }
    json root;
    try
    {
        file >> root;
    }
    catch (...)
    {
        MIRA_CHECK(false);
        return;
    }

    std::vector<std::string> options = root.at("options").get<std::vector<std::string>>();
    const std::vector<std::string>& live = mira::tamev_option_strings();
    MIRA_CHECK_EQ(options.size(), live.size());
    if (options.size() != live.size())
    {
        return;
    }
    for (std::size_t i = 0; i < live.size(); ++i)
    {
        MIRA_CHECK_EQ(options[i], live[i]);
    }

    const std::string model_path =
        mira::paths::resolve_data_path("models/model_int8.onnx");
    const std::string tok_path =
        mira::paths::resolve_data_path("models/tamev-base/tokenizer.json");
    if (!mira::paths::file_exists(model_path) || !mira::paths::file_exists(tok_path))
    {
        std::cout << "tamev_parity: model files missing, skipping\n";
        return;
    }

    mira::TamevClassifier classifier(0.0f); // no gating: compare raw argmax
    MIRA_CHECK(classifier.load_model(model_path, tok_path).is_ok());
    if (!classifier.is_loaded())
    {
        return;
    }

    int mismatches = 0;
    for (const auto& entry : root.at("vectors"))
    {
        const std::string utterance = entry.at("utterance").get<std::string>();
        std::vector<float> expected = entry.at("probs").get<std::vector<float>>();
        const int expected_argmax = entry.at("argmax").get<int>();

        mira::ClassificationResult out = classifier.classify(utterance);
        MIRA_CHECK_EQ(out.probabilities.size(), expected.size());
        if (out.probabilities.size() != expected.size())
        {
            continue;
        }
        int best = 0;
        for (std::size_t i = 1; i < out.probabilities.size(); ++i)
        {
            if (out.probabilities[i] > out.probabilities[best])
            {
                best = static_cast<int>(i);
            }
        }
        if (best != expected_argmax)
        {
            ++mismatches;
        }
        MIRA_CHECK_EQ(best, expected_argmax);
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            // int8 MatMul kernels differ between the venv ORT that generated
            // the fixture (1.30.0 via ~/v.sh) and the system ORT the C++
            // binary links (1.29.0): measured max abs diff 4.0e-04. Argmax
            // equality above is the hard gate; this guards against silent
            // large drift while tolerating known kernel skew.
            MIRA_CHECK(std::fabs(out.probabilities[i] - expected[i]) <= 1e-3f);
        }
    }
    MIRA_CHECK_EQ(mismatches, 0);
#endif
}

} // namespace

int main()
{
    parity_against_frozen_vectors();
    return mira_test::finish("tamev_parity");
}
