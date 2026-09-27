#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/intent.hpp"
#include "core/result.hpp"

namespace mira
{

// Result of one TAMEV intent classification.
struct ClassificationResult
{
    Intent intent = Intent::Unknown;
    float confidence = 0.0f;
    // One entry per candidate in tamev_option_strings(), in catalog order.
    std::vector<float> probabilities;
};

// TAMEV intent classifier: TinyBERT 4L/312D dual encoder behind ONNX Runtime.
//
// Input contract (Phase 6, see tamev/option_catalog.hpp):
//   ctx_input_ids / ctx_attention_mask      (B, ctx_len=128)      int64
//   opt_input_ids / opt_attention_mask      (B, K, opt_len=64)    int64
// Output: logits / probs                    (B, K)                float32
//
// Below the confidence threshold the classifier returns Intent::Unknown
// rather than a low-confidence wrong action (ROADMAP section 6.3).
class TamevClassifier
{
public:
    explicit TamevClassifier(float confidence_threshold = 0.5f);
    ~TamevClassifier();

    TamevClassifier(const TamevClassifier&) = delete;
    TamevClassifier& operator=(const TamevClassifier&) = delete;

    // Loads the ONNX model and the WordPiece vocabulary. Returns Unavailable
    // when files are missing/corrupt or when the build lacks ONNX Runtime, so
    // callers never crash on a bad model path.
    Result load_model(const std::string& onnx_model_path,
                      const std::string& tokenizer_json_path);

    // Classifies one utterance. Requires a prior successful load_model().
    // Never throws: failures yield Intent::Unknown with empty probabilities.
    ClassificationResult classify(const std::string& utterance) const;

    bool is_loaded() const;

    float confidence_threshold() const;
    void set_confidence_threshold(float threshold);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    float confidence_threshold_;
};

} // namespace mira
