// Unit tests for the TAMEV option catalog contract and the ONNX classifier.
//
// Model-free checks always run. Model-backed checks run only when the ONNX
// model file and tokenizer exist; otherwise the test proves the graceful
// Unavailable/Unknown fallback and skips inference.

#include "tamev/option_catalog.hpp"
#include "tamev/tamev_classifier.hpp"
#include "core/paths.hpp"
#include "test_util.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace
{

using mira::Intent;

void catalog_order_matches_training()
{
    const std::vector<std::string>& options = mira::tamev_option_strings();
    const std::vector<std::string> expected = {
        "open_application", "search_web", "read_screen", "type_text",
        "file_operation", "system_control", "conversation"};
    MIRA_CHECK_EQ(options.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        MIRA_CHECK_EQ(options[i], expected[i]);
    }
}

void catalog_maps_to_intent_catalog()
{
    const std::vector<Intent>& catalog = mira::intent_catalog();
    MIRA_CHECK_EQ(mira::tamev_option_strings().size(), catalog.size());
    for (std::size_t i = 0; i < catalog.size(); ++i)
    {
        MIRA_CHECK(mira::tamev_index_to_intent(i) == catalog[i]);
        MIRA_CHECK_EQ(mira::tamev_intent_to_index(catalog[i]), static_cast<int>(i));
    }
    MIRA_CHECK(mira::tamev_index_to_intent(999) == Intent::Unknown);
    MIRA_CHECK_EQ(mira::tamev_intent_to_index(Intent::Unknown), -1);
}

void context_format_is_canonical()
{
    MIRA_CHECK_EQ(mira::tamev_context_text("Open Firefox"),
                  std::string("Which action should Mira perform?\nUser request: Open Firefox"));
    MIRA_CHECK_EQ(mira::TAMEV_CONTEXT_LENGTH, static_cast<std::size_t>(128));
    MIRA_CHECK_EQ(mira::TAMEV_OPTION_LENGTH, static_cast<std::size_t>(64));
}

void missing_model_returns_unavailable()
{
    mira::TamevClassifier classifier;
    MIRA_CHECK(!classifier.is_loaded());
    mira::Result r = classifier.load_model("/nonexistent/model.onnx",
                                           "/nonexistent/tokenizer.json");
    MIRA_CHECK(r.is_error());
    MIRA_CHECK(!classifier.is_loaded());

    mira::ClassificationResult out = classifier.classify("Open Firefox");
    MIRA_CHECK(out.intent == Intent::Unknown);
    MIRA_CHECK_EQ(out.probabilities.size(), mira::tamev_option_strings().size());
}

void model_backed_checks()
{
    const std::string model_path =
        mira::paths::resolve_data_path("models/model_int8.onnx");
    const std::string tok_path =
        mira::paths::resolve_data_path("models/tamev-base/tokenizer.json");
    if (!mira::paths::file_exists(model_path) || !mira::paths::file_exists(tok_path))
    {
        std::cout << "tamev_classifier: model files missing, skipping inference checks\n";
        return;
    }

    mira::TamevClassifier classifier;
    mira::Result r = classifier.load_model(model_path, tok_path);
#ifdef MIRA_HAVE_ONNXRUNTIME
    MIRA_CHECK(r.is_ok());
    MIRA_CHECK(classifier.is_loaded());
    if (!classifier.is_loaded())
    {
        return;
    }

    const std::size_t k = mira::tamev_option_strings().size();
    mira::ClassificationResult out = classifier.classify("Open Firefox");
    MIRA_CHECK_EQ(out.probabilities.size(), k);
    float sum = 0.0f;
    for (float p : out.probabilities)
    {
        MIRA_CHECK(p >= 0.0f && p <= 1.0f);
        sum += p;
    }
    MIRA_CHECK(std::fabs(sum - 1.0f) < 1e-3f);

    // Confidence gate: an impossibly high threshold forces Unknown.
    classifier.set_confidence_threshold(2.0f);
    mira::ClassificationResult gated = classifier.classify("Open Firefox");
    MIRA_CHECK(gated.intent == Intent::Unknown);
    MIRA_CHECK_EQ(gated.probabilities.size(), k);
#else
    // Without ONNX Runtime the loader must report Unavailable, never crash.
    MIRA_CHECK(r.is_error());
    MIRA_CHECK(!classifier.is_loaded());
#endif
}

} // namespace

int main()
{
    catalog_order_matches_training();
    catalog_maps_to_intent_catalog();
    context_format_is_canonical();
    missing_model_returns_unavailable();
    model_backed_checks();
    return mira_test::finish("tamev_classifier");
}
