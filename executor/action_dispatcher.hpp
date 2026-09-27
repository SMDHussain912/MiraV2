#pragma once

#include <memory>
#include <unordered_map>

#include "core/action.hpp"
#include "core/intent.hpp"
#include "core/result.hpp"
#include "executor/action.hpp"

namespace mira
{

// Action Dispatcher (ROADMAP.md Section 5.8, Phase 5).
//
// Owns:
//   - Mapping ActionRequest (Intent + TokenResult) to the appropriate action handler
//   - Dispatching execution and producing a single unified Result
//   - Returning clear, well-structured errors for unknown or unsupported intents
//
// Must NOT:
//   - Contain target extraction logic (owned by Tokenizer)
//   - Contain alias resolution logic (owned by AppResolver)
//   - Launch processes or execute shell commands directly (owned by AppManager)
//   - Be a God-object
class ActionDispatcher
{
public:
    ActionDispatcher() = default;
    ~ActionDispatcher() = default;

    ActionDispatcher(const ActionDispatcher&) = delete;
    ActionDispatcher& operator=(const ActionDispatcher&) = delete;
    ActionDispatcher(ActionDispatcher&&) noexcept = default;
    ActionDispatcher& operator=(ActionDispatcher&&) noexcept = default;

    // Registers or replaces the action handler for its designated intent.
    void register_action(std::unique_ptr<IAction> action);

    // Dispatches the request to the matching action handler.
    Result dispatch(const ActionRequest& request);

    // Checks whether an action handler is currently registered for the intent.
    bool has_handler(Intent intent) const;

private:
    std::unordered_map<Intent, std::unique_ptr<IAction>> handlers_;
};

} // namespace mira
