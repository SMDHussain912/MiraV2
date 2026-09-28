#pragma once

#include <memory>
#include <string>

#include "appmgr/appmgr.hpp"
#include "core/action.hpp"
#include "core/intent.hpp"
#include "core/result.hpp"
#include "executor/action_dispatcher.hpp"
#include "resolver/app_resolver.hpp"
#include "tamev/tamev_classifier.hpp"
#include "tokenizer/tokenizer.hpp"

namespace mira
{

// Outcome of one full pipeline run: every stage's output plus the final Result.
//
// Kept as a struct (rather than only logging inside run()) so that tests and
// the CLI can assert on the individual stages of the decision chain.
struct PipelineResult
{
    std::string transcript;
    ClassificationResult classification;
    TokenResult tokens;
    // True only when an action handler was actually invoked. False whenever the
    // pipeline stopped early (empty input, confidence gate, extraction failure).
    bool dispatched = false;
    Result result;

    // Human-readable decision chain:
    //   transcript -> classify (intent + probabilities) -> extract -> dispatch -> outcome
    std::string log() const;
};

// The full runtime pipeline (ROADMAP Phase 8):
//
//   transcript -> TAMEV (Intent) -> Tokenizer (TokenResult)
//              -> ActionDispatcher -> action (resolver -> appmgr)
//
// Ownership: the pipeline owns the classifier, tokenizer, dispatcher, resolver
// and app manager so that one object is the entire post-ASR path. Tests may
// reach the individual stages through the accessors to add actions or raise
// the confidence threshold.
//
// Failure model (ROADMAP section 6.3): empty/garbage transcript and
// below-threshold classification are Rejected *before* any action can run;
// a recognised intent without an extractor/action is Unavailable, never a
// silent no-op.
class Pipeline
{
public:
    struct Config
    {
        // Empty paths resolve to the development-layout defaults:
        //   models/model_int8.onnx, models/tamev-base/tokenizer.json,
        //   config/apps.json (through mira::paths::resolve_data_path).
        std::string model_path;
        std::string tokenizer_path;
        std::string apps_config_path;
        float confidence_threshold = 0.5f;
    };

    Pipeline();
    explicit Pipeline(Config config);
    ~Pipeline() = default;

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    // Loads the ONNX model + WordPiece vocabulary and the application config,
    // then registers the built-in actions. Returns the underlying failure
    // (typically Unavailable when a file is missing) and leaves is_ready() false.
    // Idempotent: calling it again re-loads and re-registers.
    Result load();
    bool is_ready() const;

    // Runs the full chain for one transcript. Never throws; every exit path
    // fills in PipelineResult::result.
    PipelineResult run(const std::string& transcript);

    // Stage accessors (test seams / extension points).
    TamevClassifier& classifier();
    ActionDispatcher& dispatcher();
    AppManager& app_manager();

private:
    Config config_;
    TamevClassifier classifier_;
    Tokenizer tokenizer_;
    std::unique_ptr<AppResolver> resolver_; // constructed in load() from config
    AppManager app_manager_;
    ActionDispatcher dispatcher_;
    bool ready_ = false;
};

} // namespace mira
