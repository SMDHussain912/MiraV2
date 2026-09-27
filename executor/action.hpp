#pragma once

#include "core/action.hpp"
#include "core/intent.hpp"
#include "core/result.hpp"

namespace mira
{

// Abstract interface for per-intent action handlers (ROADMAP.md Section 5.8).
// Each handled intent has a dedicated class implementing this interface.
class IAction
{
public:
    virtual ~IAction() = default;

    // The single intent handled by this action.
    virtual Intent handled_intent() const = 0;

    // Executes the action request, delegating to the appropriate subsystem
    // (resolver, app manager, input automation, etc.) and returning a unified Result.
    virtual Result execute(const ActionRequest& request) = 0;
};

} // namespace mira
