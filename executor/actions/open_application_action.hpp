#pragma once

#include "appmgr/appmgr.hpp"
#include "executor/action.hpp"
#include "resolver/app_resolver.hpp"

namespace mira
{

// Action handler for Intent::OpenApplication.
// Coordinates resolving an application reference (Phase 3) and launching it (Phase 4).
class OpenApplicationAction : public IAction
{
public:
    OpenApplicationAction(AppResolver& resolver, AppManager& app_manager);
    ~OpenApplicationAction() override = default;

    Intent handled_intent() const override { return Intent::OpenApplication; }
    Result execute(const ActionRequest& request) override;

private:
    AppResolver& resolver_;
    AppManager& app_manager_;
};

} // namespace mira
