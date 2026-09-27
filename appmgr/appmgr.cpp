#include "appmgr/appmgr.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <signal.h>
#include <sstream>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace mira
{

namespace
{

bool split_command_args(const std::string& cmd, std::vector<std::string>& out_argv)
{
    out_argv.clear();
    std::string current;
    bool in_single_quote = false;
    bool in_double_quote = false;
    bool escaped = false;

    for (size_t i = 0; i < cmd.size(); ++i)
    {
        char c = cmd[i];
        if (escaped)
        {
            current += c;
            escaped = false;
            continue;
        }
        if (c == '\\' && !in_single_quote)
        {
            escaped = true;
            continue;
        }
        if (c == '\'' && !in_double_quote)
        {
            in_single_quote = !in_single_quote;
            continue;
        }
        if (c == '"' && !in_single_quote)
        {
            in_double_quote = !in_double_quote;
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(c)) && !in_single_quote && !in_double_quote)
        {
            if (!current.empty())
            {
                out_argv.push_back(current);
                current.clear();
            }
            continue;
        }
        current += c;
    }

    if (!current.empty())
    {
        out_argv.push_back(current);
    }

    if (in_single_quote || in_double_quote)
    {
        return false;
    }
    return !out_argv.empty();
}

bool find_in_path(const std::string& prog, std::string* out_full_path = nullptr)
{
    if (prog.empty())
    {
        return false;
    }
    if (prog.find('/') != std::string::npos)
    {
        if (access(prog.c_str(), X_OK) == 0)
        {
            if (out_full_path)
            {
                *out_full_path = prog;
            }
            return true;
        }
        return false;
    }
    const char* path_env = std::getenv("PATH");
    if (!path_env)
    {
        return false;
    }
    std::stringstream ss(path_env);
    std::string dir;
    while (std::getline(ss, dir, ':'))
    {
        if (dir.empty())
        {
            dir = ".";
        }
        std::filesystem::path p = std::filesystem::path(dir) / prog;
        if (access(p.c_str(), X_OK) == 0)
        {
            if (out_full_path)
            {
                *out_full_path = p.string();
            }
            return true;
        }
    }
    return false;
}

bool find_desktop_file(const std::string& desktop_file, std::string* out_path = nullptr)
{
    if (desktop_file.empty())
    {
        return false;
    }
    if (desktop_file.find('/') != std::string::npos)
    {
        if (access(desktop_file.c_str(), R_OK) == 0)
        {
            if (out_path)
            {
                *out_path = desktop_file;
            }
            return true;
        }
        return false;
    }
    std::string candidate_name = desktop_file;
    if (candidate_name.size() < 8 || candidate_name.substr(candidate_name.size() - 8) != ".desktop")
    {
        candidate_name += ".desktop";
    }

    std::vector<std::string> search_dirs;
    const char* xdg_data_home = std::getenv("XDG_DATA_HOME");
    if (xdg_data_home && *xdg_data_home)
    {
        search_dirs.push_back(std::string(xdg_data_home) + "/applications");
    }
    else
    {
        const char* home = std::getenv("HOME");
        if (home && *home)
        {
            search_dirs.push_back(std::string(home) + "/.local/share/applications");
        }
    }
    const char* xdg_data_dirs = std::getenv("XDG_DATA_DIRS");
    if (xdg_data_dirs && *xdg_data_dirs)
    {
        std::stringstream ss(xdg_data_dirs);
        std::string dir;
        while (std::getline(ss, dir, ':'))
        {
            if (!dir.empty())
            {
                search_dirs.push_back(dir + "/applications");
            }
        }
    }
    else
    {
        search_dirs.push_back("/usr/local/share/applications");
        search_dirs.push_back("/usr/share/applications");
    }

    for (const auto& dir : search_dirs)
    {
        std::filesystem::path p = std::filesystem::path(dir) / candidate_name;
        if (access(p.c_str(), R_OK) == 0)
        {
            if (out_path)
            {
                *out_path = p.string();
            }
            return true;
        }
    }
    return false;
}

Result validate_working_directory(const std::string& working_dir)
{
    if (working_dir.empty())
    {
        return Result::ok();
    }
    struct stat st;
    if (stat(working_dir.c_str(), &st) != 0)
    {
        return Result::rejected("working directory does not exist: " + working_dir);
    }
    if (!S_ISDIR(st.st_mode))
    {
        return Result::rejected("working directory is not a directory: " + working_dir);
    }
    if (access(working_dir.c_str(), X_OK) != 0)
    {
        return Result::rejected("working directory is not accessible (permission denied): " + working_dir);
    }
    return Result::ok();
}

Result prepare_argv(const LaunchSpec& spec, std::vector<std::string>& out_argv)
{
    out_argv.clear();
    switch (spec.method)
    {
        case LaunchMethod::Command:
        {
            if (spec.command.empty())
            {
                return Result::rejected("command is empty");
            }
            if (!split_command_args(spec.command, out_argv))
            {
                return Result::rejected("malformed command string or unclosed quotes: " + spec.command);
            }
            if (!find_in_path(out_argv[0]))
            {
                return Result::rejected("executable not found in PATH: " + out_argv[0]);
            }
            return Result::ok();
        }
        case LaunchMethod::Executable:
        {
            if (spec.path.empty())
            {
                return Result::rejected("executable path is empty");
            }
            if (access(spec.path.c_str(), F_OK) != 0)
            {
                return Result::rejected("executable does not exist: " + spec.path);
            }
            if (access(spec.path.c_str(), X_OK) != 0)
            {
                return Result::rejected("path is not executable: " + spec.path);
            }
            out_argv.push_back(spec.path);
            return Result::ok();
        }
        case LaunchMethod::Script:
        {
            if (spec.path.empty())
            {
                return Result::rejected("script path is empty");
            }
            if (access(spec.path.c_str(), F_OK) != 0)
            {
                return Result::rejected("script does not exist: " + spec.path);
            }
            if (access(spec.path.c_str(), R_OK) != 0)
            {
                return Result::rejected("script is not readable: " + spec.path);
            }
            if (access(spec.path.c_str(), X_OK) == 0)
            {
                out_argv.push_back(spec.path);
            }
            else
            {
                out_argv.push_back("/bin/sh");
                out_argv.push_back(spec.path);
            }
            return Result::ok();
        }
        case LaunchMethod::AppImage:
        {
            if (spec.path.empty())
            {
                return Result::rejected("appimage path is empty");
            }
            if (access(spec.path.c_str(), F_OK) != 0)
            {
                return Result::rejected("appimage does not exist: " + spec.path);
            }
            if (access(spec.path.c_str(), X_OK) != 0)
            {
                return Result::rejected("appimage is not executable: " + spec.path);
            }
            out_argv.push_back(spec.path);
            return Result::ok();
        }
        case LaunchMethod::Desktop:
        {
            if (spec.desktop_file.empty())
            {
                return Result::rejected("desktop_file is empty");
            }
            std::string resolved_path;
            if (!find_desktop_file(spec.desktop_file, &resolved_path))
            {
                return Result::rejected("desktop file not found: " + spec.desktop_file);
            }
            if (find_in_path("gtk-launch"))
            {
                std::string desktop_id = spec.desktop_file;
                size_t slash = desktop_id.rfind('/');
                if (slash != std::string::npos)
                {
                    desktop_id = desktop_id.substr(slash + 1);
                }
                out_argv.push_back("gtk-launch");
                out_argv.push_back(desktop_id);
                return Result::ok();
            }
            if (find_in_path("gio"))
            {
                out_argv.push_back("gio");
                out_argv.push_back("launch");
                out_argv.push_back(resolved_path);
                return Result::ok();
            }
            if (find_in_path("xdg-open"))
            {
                out_argv.push_back("xdg-open");
                out_argv.push_back(resolved_path);
                return Result::ok();
            }
            return Result::rejected("no suitable desktop launcher found (gtk-launch, gio, xdg-open)");
        }
    }
    return Result::rejected("unknown launch method");
}

} // namespace

Result AppManager::launch_dry_run(const LaunchSpec& spec) const
{
    Result wd_res = validate_working_directory(spec.working_directory);
    if (!wd_res.is_ok())
    {
        return wd_res;
    }

    std::vector<std::string> argv;
    Result prep_res = prepare_argv(spec, argv);
    if (!prep_res.is_ok())
    {
        return prep_res;
    }

    return Result::ok();
}

Result AppManager::launch(const LaunchSpec& spec, ProcessHandle* out_handle)
{
    Result dry_res = launch_dry_run(spec);
    if (!dry_res.is_ok())
    {
        return dry_res;
    }

    std::vector<std::string> argv;
    Result prep_res = prepare_argv(spec, argv);
    if (!prep_res.is_ok())
    {
        return prep_res;
    }

    std::vector<char*> c_argv;
    c_argv.reserve(argv.size() + 1);
    for (auto& arg : argv)
    {
        c_argv.push_back(arg.data());
    }
    c_argv.push_back(nullptr);

    pid_t pid = fork();
    if (pid < 0)
    {
        int err = errno;
        return Result::failed("fork failed", std::strerror(err));
    }

    if (pid == 0)
    {
        setsid();

        if (!spec.working_directory.empty())
        {
            if (chdir(spec.working_directory.c_str()) != 0)
            {
                _exit(126);
            }
        }

        int devnull = open("/dev/null", O_RDWR);
        if (devnull >= 0)
        {
            dup2(devnull, STDIN_FILENO);
            dup2(devnull, STDOUT_FILENO);
            dup2(devnull, STDERR_FILENO);
            if (devnull > STDERR_FILENO)
            {
                close(devnull);
            }
        }

        execvp(c_argv[0], c_argv.data());
        _exit(127);
    }

    ProcessHandle handle;
    handle.pid = pid;
    handle.app_name = spec.name.empty() ? argv[0] : spec.name;
    handle.method = spec.method;

    tracked_processes_.push_back(handle);

    if (out_handle)
    {
        *out_handle = handle;
    }

    return Result::ok();
}

std::vector<pid_t> AppManager::running_pids() const
{
    std::vector<pid_t> pids;
    pids.reserve(tracked_processes_.size());
    for (const auto& handle : tracked_processes_)
    {
        pids.push_back(handle.pid);
    }
    return pids;
}

void AppManager::reap_children()
{
    int status = 0;
    for (auto it = tracked_processes_.begin(); it != tracked_processes_.end();)
    {
        pid_t res = waitpid(it->pid, &status, WNOHANG);
        if (res > 0 || (res < 0 && errno == ECHILD))
        {
            it = tracked_processes_.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

bool AppManager::is_running(pid_t pid) const
{
    if (pid <= 0)
    {
        return false;
    }

    if (kill(pid, 0) == 0)
    {
        return true;
    }

    return (errno == EPERM);
}

void AppManager::clear_tracked()
{
    tracked_processes_.clear();
}

} // namespace mira

