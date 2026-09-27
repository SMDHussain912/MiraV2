#include "appmgr/appmgr.hpp"
#include "resolver/launch_spec.hpp"

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#define TEST_ASSERT(cond)                                                       \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::cerr << "Assertion failed: " #cond                             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

namespace
{

void test_dry_run_command_valid()
{
    mira::AppManager mgr;
    mira::LaunchSpec spec;
    spec.name = "TrueCommand";
    spec.method = mira::LaunchMethod::Command;
    spec.command = "true --help";

    mira::Result res = mgr.launch_dry_run(spec);
    TEST_ASSERT(res.is_ok());
}

void test_dry_run_command_missing_executable()
{
    mira::AppManager mgr;
    mira::LaunchSpec spec;
    spec.name = "NonExistent";
    spec.method = mira::LaunchMethod::Command;
    spec.command = "non_existent_binary_xyz_12345 --arg";

    mira::Result res = mgr.launch_dry_run(spec);
    TEST_ASSERT(!res.is_ok());
    TEST_ASSERT(res.status() == mira::ResultStatus::Rejected);
}

void test_dry_run_executable_valid_and_invalid()
{
    mira::AppManager mgr;
    mira::LaunchSpec spec;
    spec.name = "TrueExec";
    spec.method = mira::LaunchMethod::Executable;
    spec.path = "/usr/bin/true";

    mira::Result res = mgr.launch_dry_run(spec);
    TEST_ASSERT(res.is_ok());

    // Non-existent path
    spec.path = "/usr/bin/does_not_exist_xyz_123";
    res = mgr.launch_dry_run(spec);
    TEST_ASSERT(!res.is_ok());

    // Not executable (e.g. /etc/hosts)
    spec.path = "/etc/hosts";
    res = mgr.launch_dry_run(spec);
    TEST_ASSERT(!res.is_ok());
}

void test_dry_run_working_directory_validation()
{
    mira::AppManager mgr;
    mira::LaunchSpec spec;
    spec.name = "WithDir";
    spec.method = mira::LaunchMethod::Command;
    spec.command = "true";
    spec.working_directory = "/tmp";

    mira::Result res = mgr.launch_dry_run(spec);
    TEST_ASSERT(res.is_ok());

    // Non-existent working directory
    spec.working_directory = "/tmp/dir_that_does_not_exist_abc_987";
    res = mgr.launch_dry_run(spec);
    TEST_ASSERT(!res.is_ok());
    TEST_ASSERT(res.status() == mira::ResultStatus::Rejected);
}

void test_dry_run_script()
{
    mira::AppManager mgr;

    // Create a temporary script file
    std::string script_path = "/tmp/mira_test_script.sh";
    {
        std::ofstream ofs(script_path);
        ofs << "#!/bin/sh\nexit 0\n";
    }

    mira::LaunchSpec spec;
    spec.name = "TestScript";
    spec.method = mira::LaunchMethod::Script;
    spec.path = script_path;

    mira::Result res = mgr.launch_dry_run(spec);
    TEST_ASSERT(res.is_ok());

    std::filesystem::remove(script_path);

    // After deletion, dry run should fail
    res = mgr.launch_dry_run(spec);
    TEST_ASSERT(!res.is_ok());
}

void test_dry_run_appimage()
{
    mira::AppManager mgr;
    std::string appimage_path = "/tmp/mira_test_fake.AppImage";
    {
        std::ofstream ofs(appimage_path);
        ofs << "dummy binary";
    }
    // Make it executable
    ::chmod(appimage_path.c_str(), 0755);

    mira::LaunchSpec spec;
    spec.name = "FakeAppImage";
    spec.method = mira::LaunchMethod::AppImage;
    spec.path = appimage_path;

    mira::Result res = mgr.launch_dry_run(spec);
    TEST_ASSERT(res.is_ok());

    std::filesystem::remove(appimage_path);

    // After removal
    res = mgr.launch_dry_run(spec);
    TEST_ASSERT(!res.is_ok());
}

void test_dry_run_desktop()
{
    mira::AppManager mgr;
    mira::LaunchSpec spec;
    spec.name = "DesktopApp";
    spec.method = mira::LaunchMethod::Desktop;

    // A desktop file that does not exist
    spec.desktop_file = "completely_imaginary_app_xyz.desktop";
    mira::Result res = mgr.launch_dry_run(spec);
    TEST_ASSERT(!res.is_ok());
}

void test_live_launch_and_process_tracking()
{
    mira::AppManager mgr;
    mira::LaunchSpec spec;
    spec.name = "SleepCommand";
    spec.method = mira::LaunchMethod::Command;
    spec.command = "sleep 0.1";

    mira::ProcessHandle handle;
    mira::Result res = mgr.launch(spec, &handle);
    TEST_ASSERT(res.is_ok());
    TEST_ASSERT(handle.pid > 0);
    TEST_ASSERT(handle.app_name == "SleepCommand");
    TEST_ASSERT(handle.method == mira::LaunchMethod::Command);

    // Process should be in running_pids
    std::vector<pid_t> pids = mgr.running_pids();
    TEST_ASSERT(!pids.empty());
    TEST_ASSERT(pids[0] == handle.pid);

    // Wait for the sleep child to finish
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    // Reaping children cleans up the terminated process
    mgr.reap_children();
    TEST_ASSERT(!mgr.is_running(handle.pid));
    pids = mgr.running_pids();
    TEST_ASSERT(pids.empty());
}

} // namespace

int main()
{
    test_dry_run_command_valid();
    test_dry_run_command_missing_executable();
    test_dry_run_executable_valid_and_invalid();
    test_dry_run_working_directory_validation();
    test_dry_run_script();
    test_dry_run_appimage();
    test_dry_run_desktop();
    test_live_launch_and_process_tracking();

    std::cout << "All appmgr tests passed successfully." << std::endl;
    return 0;
}
