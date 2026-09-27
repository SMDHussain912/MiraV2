#include "core/paths.hpp"

#include <cstdlib>
#include <sys/stat.h>

#include <limits.h>
#include <unistd.h>

namespace mira
{
namespace paths
{

namespace
{

bool is_absolute(const std::string& path)
{
    return !path.empty() && path.front() == '/';
}

} // namespace

std::string executable_directory()
{
    char buffer[PATH_MAX];

    const ssize_t length = ::readlink(
        "/proc/self/exe",
        buffer,
        sizeof(buffer) - 1
    );

    if (length <= 0)
    {
        return {};
    }

    buffer[length] = '\0';

    const std::string path(buffer);

    const std::size_t separator = path.find_last_of('/');

    if (separator == std::string::npos)
    {
        return {};
    }

    return path.substr(0, separator);
}

std::string source_root()
{
#ifdef MIRA_SOURCE_ROOT
    return MIRA_SOURCE_ROOT;
#else
    return {};
#endif
}

std::string data_directory()
{
    const char* override_dir = std::getenv("MIRA_DATA_DIR");

    if (override_dir != nullptr && *override_dir != '\0')
    {
        return override_dir;
    }

    const std::string executable_dir = executable_directory();

    if (!executable_dir.empty())
    {
        const std::string installed_candidate = executable_dir + "/data";

        if (is_directory(installed_candidate))
        {
            return installed_candidate;
        }
    }

    const std::string root = source_root();

    if (!root.empty() && is_directory(root))
    {
        return root;
    }

    return ".";
}

std::string resolve_data_path(const std::string& path)
{
    if (is_absolute(path))
    {
        return path;
    }

    return data_directory() + "/" + path;
}

bool file_exists(const std::string& path)
{
    struct stat status;

    if (::stat(path.c_str(), &status) != 0)
    {
        return false;
    }

    return S_ISREG(status.st_mode);
}

bool is_directory(const std::string& path)
{
    struct stat status;

    if (::stat(path.c_str(), &status) != 0)
    {
        return false;
    }

    return S_ISDIR(status.st_mode);
}

} // namespace paths
} // namespace mira
