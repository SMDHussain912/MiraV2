#pragma once

#include <string>
#include <vector>
#include <sys/types.h>

#include "core/result.hpp"
#include "resolver/launch_spec.hpp"

namespace mira
{

// Records a successfully spawned application instance.
struct ProcessHandle
{
    pid_t pid = -1;
    std::string app_name;
    LaunchMethod method = LaunchMethod::Command;
};

// Application Manager (ROADMAP.md Section 5.7, Phase 4).
//
// Owns:
//   - Executing a LaunchSpec using the launch method it specifies
//   - Honouring working_directory
//   - Tracking child process IDs
//   - Reporting success or failure with OS error diagnostics
//   - Safe, dry-run validation mode for testing without launching real windows
//
// Must NOT:
//   - Search for applications
//   - Read apps.json
//   - Guess aliases or classify intent
//   - Run arbitrary shell strings through /bin/sh -c
class AppManager
{
public:
    AppManager() = default;
    ~AppManager() = default;

    // Disallow copying to avoid ambiguous child tracking ownership
    AppManager(const AppManager&) = delete;
    AppManager& operator=(const AppManager&) = delete;
    AppManager(AppManager&&) noexcept = default;
    AppManager& operator=(AppManager&&) noexcept = default;

    // Executes the LaunchSpec directly (non-shell). If successful, tracks the child PID.
    Result launch(const LaunchSpec& spec, ProcessHandle* out_handle = nullptr);

    // Dry-run mode: performs all pre-flight binary, path, and working directory
    // validations without spawning any processes.
    Result launch_dry_run(const LaunchSpec& spec) const;

    // Returns all tracked child PIDs.
    std::vector<pid_t> running_pids() const;

    // Reaps zombie children and prunes terminated processes from tracking.
    void reap_children();

    // Checks if a tracked process is still alive.
    bool is_running(pid_t pid) const;

    // Explicitly clear all tracked processes (for teardown/testing).
    void clear_tracked();

private:
    std::vector<ProcessHandle> tracked_processes_;
};

} // namespace mira

