// End-to-end pipeline tests (ROADMAP Phase 8): transcript -> TAMEV -> Tokenizer
// -> ActionDispatcher -> action, driven by TEXT INPUT so they bypass the
// microphone and stay deterministic/CI-friendly.
//
// The model-dependent cases skip (not fail) when the ONNX artifact is absent,
// matching the behaviour of tests/test_tamev_parity.cpp: the suite must be
// runnable without a model file.

#include "core/paths.hpp"
#include "pipeline/pipeline.hpp"
#include "test_util.hpp"

#include <fstream>
#include <string>

namespace
{

// Fixture config: Firefox maps to a harmless short-lived command so the happy
// path exercises a REAL launch without touching the user's applications.
const char* kFixtureAppsJson = R"({
    "version": 1,
    "applications": [
        {
            "name": "Firefox",
            "aliases": ["firefox", "mozilla firefox"],
            "launch": { "method": "command", "command": "sleep 0.1" }
        }
    ]
})";

std::string write_fixture_config()
{
    const std::string path = "/tmp/mira_test_pipeline_apps.json";
    std::ofstream file(path);
    file << kFixtureAppsJson;
    return path;
}

void test_empty_transcript_never_reaches_tamev(mira::Pipeline& pipeline)
{
    mira::PipelineResult r = pipeline.run("");
    MIRA_CHECK(r.classification.intent == mira::Intent::Unknown);
    MIRA_CHECK(r.classification.probabilities.empty()); // no TAMEV call
    MIRA_CHECK(!r.dispatched);
    MIRA_CHECK(r.result.status() == mira::ResultStatus::Rejected);
    MIRA_CHECK(r.result.message() == "I didn't catch that");

    r = pipeline.run("   \t  ");
    MIRA_CHECK(!r.dispatched);
    MIRA_CHECK(r.result.status() == mira::ResultStatus::Rejected);
}

void test_low_confidence_never_dispatches(const mira::Pipeline::Config& base)
{
    // Threshold above any achievable confidence makes every classification a
    // low-confidence one: the gate must hold regardless of model internals.
    mira::Pipeline::Config gated = base;
    gated.confidence_threshold = 1.01f;

    mira::Pipeline pipeline(gated);
    MIRA_CHECK(pipeline.load().is_ok());

    mira::PipelineResult r = pipeline.run("Open Firefox");
    MIRA_CHECK(r.classification.intent == mira::Intent::Unknown);
    MIRA_CHECK(!r.dispatched);
    MIRA_CHECK(r.result.status() == mira::ResultStatus::Rejected);
    MIRA_CHECK(r.result.message() == "I didn't catch that");
    MIRA_CHECK(r.log().find("dispatch:   skipped") != std::string::npos);
}

void test_open_application_happy_path(mira::Pipeline& pipeline)
{
    mira::PipelineResult r = pipeline.run("Open Firefox");

    // Stage-by-stage: classification, extraction, dispatch, outcome.
    MIRA_CHECK(r.classification.intent == mira::Intent::OpenApplication);
    MIRA_CHECK(r.tokens.valid);
    MIRA_CHECK(r.tokens.type == mira::TokenType::TARGET);
    MIRA_CHECK(r.tokens.value == "Firefox");
    MIRA_CHECK(r.dispatched);
    MIRA_CHECK(r.result.is_ok());

    // The decision chain must be complete and in order.
    const std::string log = r.log();
    const std::size_t transcript_at = log.find("transcript:");
    const std::size_t classify_at = log.find("classify:");
    const std::size_t extract_at = log.find("extract:");
    const std::size_t dispatch_at = log.find("dispatch:");
    const std::size_t outcome_at = log.find("outcome:");
    MIRA_CHECK(transcript_at != std::string::npos);
    MIRA_CHECK(classify_at != std::string::npos);
    MIRA_CHECK(extract_at != std::string::npos);
    MIRA_CHECK(dispatch_at != std::string::npos);
    MIRA_CHECK(outcome_at != std::string::npos);
    MIRA_CHECK(transcript_at < classify_at);
    MIRA_CHECK(classify_at < extract_at);
    MIRA_CHECK(extract_at < dispatch_at);
    MIRA_CHECK(dispatch_at < outcome_at);

    pipeline.app_manager().reap_children();
}

void test_unconfigured_target_is_rejected(mira::Pipeline& pipeline)
{
    mira::PipelineResult r = pipeline.run("Open Minecraft");

    MIRA_CHECK(r.dispatched);     // extraction succeeded, action attempted
    MIRA_CHECK(!r.result.is_ok()); // but resolution refused it
    // When the classifier agrees this is open_application (expected), the
    // failure must be the resolver's honest "not configured" rejection.
    if (r.classification.intent == mira::Intent::OpenApplication)
    {
        MIRA_CHECK(r.result.status() == mira::ResultStatus::Rejected);
        MIRA_CHECK(r.result.message() == "not configured");
    }
}

void test_unsupported_intent_is_unavailable_not_action(mira::Pipeline& pipeline)
{
    mira::PipelineResult r = pipeline.run("Turn the volume down");

    MIRA_CHECK(!r.result.is_ok());
    MIRA_CHECK(!r.dispatched); // no extractor -> nothing may run
    if (r.classification.intent == mira::Intent::SystemControl)
    {
        MIRA_CHECK(r.result.status() == mira::ResultStatus::Unavailable);
        MIRA_CHECK(r.result.message() == "intent not supported yet");
    }
}

} // namespace

int main()
{
    const std::string fixture = write_fixture_config();

    mira::Pipeline::Config config;
    config.apps_config_path = fixture; // model paths use the defaults

    mira::Pipeline pipeline(config);
    const mira::Result loaded = pipeline.load();
    if (!loaded.is_ok())
    {
        std::cout << "test_pipeline: model unavailable, skipping ("
                  << loaded.message() << ")\n";
        return mira_test::finish("pipeline");
    }

    test_empty_transcript_never_reaches_tamev(pipeline);
    test_low_confidence_never_dispatches(config);
    test_open_application_happy_path(pipeline);
    test_unconfigured_target_is_rejected(pipeline);
    test_unsupported_intent_is_unavailable_not_action(pipeline);

    return mira_test::finish("pipeline");
}
