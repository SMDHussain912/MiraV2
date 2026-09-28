#include "pipeline/pipeline.hpp"

#include "core/paths.hpp"
#include "core/text_utils.hpp"
#include "executor/actions/open_application_action.hpp"
#include "resolver/application_config.hpp"

#include <iomanip>
#include <sstream>
#include <utility>

namespace mira
{

namespace
{

// Default data-relative paths (development and installed layouts both resolve
// through mira::paths::resolve_data_path).
const char* const kDefaultModelPath = "models/training/export/model_fp32.onnx";
const char* const kDefaultTokenizerPath = "models/tamev-base/tokenizer.json";
const char* const kDefaultAppsConfigPath = "config/apps.json";

std::string format_confidence(float value)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(4) << value;
    return out.str();
}

} // namespace

std::string PipelineResult::log() const
{
    std::ostringstream out;
    out << "transcript: \"" << transcript << "\"\n";

    out << "classify:   " << to_string(classification.intent)
        << " conf=" << format_confidence(classification.confidence);
    if (!classification.probabilities.empty())
    {
        out << " probs=[";
        for (std::size_t i = 0; i < classification.probabilities.size(); ++i)
        {
            if (i > 0)
            {
                out << ',';
            }
            out << classification.probabilities[i];
        }
        out << ']';
    }
    out << '\n';

    if (tokens.status == TokenStatus::EmptyInput ||
        tokens.status == TokenStatus::UnsupportedIntent)
    {
        out << "extract:    skipped (" << to_string(tokens.status) << ")\n";
    }
    else
    {
        out << "extract:    " << to_string(tokens.type) << "=\"" << tokens.value
            << "\" " << to_string(tokens.status)
            << (tokens.valid ? " valid" : " invalid") << '\n';
    }

    out << "dispatch:   " << (dispatched ? "attempted" : "skipped") << '\n';
    out << "outcome:    " << result.to_string() << '\n';
    return out.str();
}

Pipeline::Pipeline()
    : Pipeline(Config{})
{
}

Pipeline::Pipeline(Config config)
    : config_(std::move(config))
    , classifier_(config_.confidence_threshold)
{
}

Result Pipeline::load()
{
    ready_ = false;

    const std::string model_path = config_.model_path.empty()
        ? paths::resolve_data_path(kDefaultModelPath)
        : config_.model_path;
    const std::string tokenizer_path = config_.tokenizer_path.empty()
        ? paths::resolve_data_path(kDefaultTokenizerPath)
        : config_.tokenizer_path;
    const std::string apps_config_path = config_.apps_config_path.empty()
        ? paths::resolve_data_path(kDefaultAppsConfigPath)
        : config_.apps_config_path;

    Result model_res = classifier_.load_model(model_path, tokenizer_path);
    if (!model_res.is_ok())
    {
        return model_res;
    }

    ApplicationConfig config;
    Result config_res = ApplicationConfig::load_from_file(apps_config_path, config);
    if (!config_res.is_ok())
    {
        return config_res;
    }

    resolver_ = std::make_unique<AppResolver>(std::move(config));

    // Built-in actions. Each intent gets its own handler as it is brought up
    // (ROADMAP Phase 8 task 4): nothing beyond open_application exists yet.
    dispatcher_.register_action(
        std::make_unique<OpenApplicationAction>(*resolver_, app_manager_));

    ready_ = true;
    return Result::ok();
}

bool Pipeline::is_ready() const
{
    return ready_;
}

PipelineResult Pipeline::run(const std::string& transcript)
{
    PipelineResult out;
    out.transcript = transcript;

    if (!ready_)
    {
        out.result = Result::unavailable("pipeline not loaded",
                                         "Pipeline::load() must succeed first");
        return out;
    }

    const std::string trimmed = trim(transcript);
    if (trimmed.empty())
    {
        // ROADMAP 6.3: empty/garbage transcript -> Rejected, no TAMEV call.
        out.result = Result::rejected("I didn't catch that",
                                      "empty transcript; TAMEV not called");
        return out;
    }

    out.classification = classifier_.classify(trimmed);
    if (out.classification.intent == Intent::Unknown)
    {
        // ROADMAP 6.3: confidence below threshold -> ask, do not act.
        out.result = Result::rejected(
            "I didn't catch that",
            "confidence " + format_confidence(out.classification.confidence) +
                " below threshold " +
                format_confidence(classifier_.confidence_threshold()) +
                "; clarification required");
        return out;
    }

    out.tokens = tokenizer_.process(trimmed, out.classification.intent);
    if (!out.tokens.valid)
    {
        // Never dispatch an unextracted request (ROADMAP 6.3: tokenizer
        // failure -> Rejected/Unavailable, no resolver call).
        switch (out.tokens.status)
        {
        case TokenStatus::MissingTarget:
            // The only extractor today is open_application, so the
            // clarification prompt is application-shaped.
            out.result = Result::rejected("what should I open?",
                                          "tokenizer returned no target");
            break;
        case TokenStatus::UnsupportedIntent:
            out.result = Result::unavailable(
                "intent not supported yet",
                "no tokenizer extractor for " +
                    to_string(out.classification.intent));
            break;
        case TokenStatus::EmptyInput:
            out.result = Result::rejected("I didn't catch that",
                                          "tokenizer received empty input");
            break;
        default:
            out.result = Result::rejected("could not extract a target",
                                          "TokenStatus: " +
                                              to_string(out.tokens.status));
            break;
        }
        return out;
    }

    ActionRequest request;
    request.intent = out.classification.intent;
    request.tokens = out.tokens;
    request.original_text = trimmed;

    out.dispatched = true;
    out.result = dispatcher_.dispatch(request);
    return out;
}

TamevClassifier& Pipeline::classifier()
{
    return classifier_;
}

ActionDispatcher& Pipeline::dispatcher()
{
    return dispatcher_;
}

AppManager& Pipeline::app_manager()
{
    return app_manager_;
}

} // namespace mira
