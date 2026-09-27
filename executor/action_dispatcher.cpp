#include "executor/action_dispatcher.hpp"

namespace mira
{

void ActionDispatcher::register_action(std::unique_ptr<IAction> action)
{
    if (!action)
    {
        return;
    }
    Intent intent = action->handled_intent();
    handlers_[intent] = std::move(action);
}

Result ActionDispatcher::dispatch(const ActionRequest& request)
{
    if (request.intent == Intent::Unknown)
    {
        return Result::rejected(
            "unrecognized intent",
            "Intent::Unknown cannot be executed; clarification or re-prompt required");
    }

    auto it = handlers_.find(request.intent);
    if (it == handlers_.end() || !it->second)
    {
        return Result::unavailable(
            "intent not supported yet",
            "No action handler registered for intent: " + to_string(request.intent));
    }

    return it->second->execute(request);
}

bool ActionDispatcher::has_handler(Intent intent) const
{
    auto it = handlers_.find(intent);
    return (it != handlers_.end() && it->second != nullptr);
}

} // namespace mira
