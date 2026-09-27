#pragma once

#include <string>

namespace mira
{
namespace paths
{

// Directory containing the running executable, resolved through /proc/self/exe.
// Empty when it cannot be determined.
std::string executable_directory();

// Root of the MiraV2 source tree that this binary was built from. It is baked in
// at configure time (MIRA_SOURCE_ROOT) so that development-time data files are
// found regardless of the working directory the binary is started from.
// Empty when the build did not define it.
std::string source_root();

// Directory that data files (models, configuration) are resolved against:
//   1. $MIRA_DATA_DIR when set and non-empty
//   2. <executable directory>/data when that directory exists (installed layout)
//   3. source_root() when it is a directory (development layout)
//   4. "." as a last resort
std::string data_directory();

// Converts a data-relative path into an absolute path. An absolute input path is
// returned unchanged.
std::string resolve_data_path(const std::string& path);

bool file_exists(const std::string& path);
bool is_directory(const std::string& path);

} // namespace paths
} // namespace mira
