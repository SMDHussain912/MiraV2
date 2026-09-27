#pragma once

#include <string>

namespace mira
{

// How an application should be launched on Linux.
//
// Documented in ROADMAP.md Section 5.6 and Section 5.7:
//   Command:    A program resolved through PATH (with optional arguments)
//   Executable: An absolute path to a binary executable
//   Script:     A script at an absolute path (invoked with suitable shell/interpreter)
//   AppImage:   A self-contained AppImage bundle
//   Desktop:    A .desktop entry resolved through XDG specifications
enum class LaunchMethod
{
    Command,
    Executable,
    Script,
    AppImage,
    Desktop
};

std::string to_string(LaunchMethod method);
LaunchMethod launch_method_from_string(const std::string& str);

// Pure description of how an application is to be launched.
//
// This is the handoff structure between the Resolver (Phase 3) and the
// AppManager (Phase 4). It is a pure value type: it contains no process handles,
// no execution methods, and causes zero OS side effects.
struct LaunchSpec
{
    std::string name;                          // Canonical application name
    LaunchMethod method = LaunchMethod::Command;
    std::string command;                       // Non-empty when method == Command
    std::string path;                          // Non-empty when method in {Executable, Script, AppImage}
    std::string desktop_file;                  // Non-empty when method == Desktop
    std::string working_directory;             // Optional (empty if none)
};

} // namespace mira
