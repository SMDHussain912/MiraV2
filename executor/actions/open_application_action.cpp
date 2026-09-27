#include "executor/actions/open_application_action.hpp"

namespace mira
{

OpenApplicationAction::OpenApplicationAction(AppResolver& resolver, AppManager& app_manager)
    : resolver_(resolver)
    , app_manager_(app_manager)
{
}

Result OpenApplicationAction::execute(const ActionRequest& request)
{
    if (request.intent != Intent::OpenApplication)
    {
        return Result::rejected(
            "mismatched action intent",
            "OpenApplicationAction received intent: " + to_string(request.intent));
    }

    if (!request.tokens.valid || request.tokens.status != TokenStatus::Success)
    {
        std::string detail = "TokenStatus: " + to_string(request.tokens.status);
        return Result::rejected("missing or invalid application target", detail);
    }

    if (request.tokens.value.empty())
    {
        return Result::rejected("empty application target");
    }

    // Phase 3: Resolve the target into a pure LaunchSpec
    LaunchSpec spec;
    Result resolve_res = resolver_.resolve(request.tokens, spec);
    if (!resolve_res.is_ok())
    {
        return resolve_res;
    }

    // Phase 4: Execute the resolved spec
    return app_manager_.launch(spec);
}

} // namespace mira
