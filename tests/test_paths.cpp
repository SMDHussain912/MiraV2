// Unit tests for data path resolution (core/paths.*).
//
// The point of this module is that model and configuration files are found
// regardless of the working directory the binary was started from, and that the
// location can be overridden explicitly.

#include "core/paths.hpp"
#include "test_util.hpp"

#include <cstdlib>
#include <string>

namespace
{

void executable_directory_is_absolute()
{
    const std::string directory = mira::paths::executable_directory();

    MIRA_CHECK(!directory.empty());
    MIRA_CHECK(directory.front() == '/');
}

void source_root_points_at_the_source_tree()
{
    const std::string root = mira::paths::source_root();

    MIRA_CHECK(!root.empty());
    MIRA_CHECK(root.front() == '/');

    // Proves the compile-time definition really points at the project tree.
    MIRA_CHECK(mira::paths::file_exists(root + "/CMakeLists.txt"));
    MIRA_CHECK(mira::paths::is_directory(root + "/core"));
}

void the_environment_override_wins()
{
    ::setenv("MIRA_DATA_DIR", "/tmp/mira-test-data-dir", 1);

    MIRA_CHECK_EQ(
        mira::paths::data_directory(),
        std::string("/tmp/mira-test-data-dir")
    );

    ::unsetenv("MIRA_DATA_DIR");
}

void the_fallback_is_a_real_absolute_directory()
{
    ::unsetenv("MIRA_DATA_DIR");

    const std::string directory = mira::paths::data_directory();

    MIRA_CHECK(!directory.empty());
    MIRA_CHECK(directory.front() == '/');
    MIRA_CHECK(mira::paths::is_directory(directory));
}

void relative_paths_are_resolved_against_the_data_directory()
{
    ::setenv("MIRA_DATA_DIR", "/tmp/mira-test-data-dir", 1);

    MIRA_CHECK_EQ(
        mira::paths::resolve_data_path("models/ggml-base.en.bin"),
        std::string("/tmp/mira-test-data-dir/models/ggml-base.en.bin")
    );

    ::unsetenv("MIRA_DATA_DIR");
}

void absolute_paths_pass_through_unchanged()
{
    ::unsetenv("MIRA_DATA_DIR");

    MIRA_CHECK_EQ(
        mira::paths::resolve_data_path("/usr/share/mira/model.bin"),
        std::string("/usr/share/mira/model.bin")
    );
}

void file_and_directory_checks()
{
    const std::string root = mira::paths::source_root();

    MIRA_CHECK(mira::paths::file_exists(root + "/CMakeLists.txt"));
    MIRA_CHECK(!mira::paths::is_directory(root + "/CMakeLists.txt"));
    MIRA_CHECK(!mira::paths::file_exists(root + "/mira-no-such-file-xyz"));
    MIRA_CHECK(!mira::paths::is_directory(root + "/mira-no-such-directory-xyz"));
}

} // namespace

int main()
{
    executable_directory_is_absolute();
    source_root_points_at_the_source_tree();
    the_environment_override_wins();
    the_fallback_is_a_real_absolute_directory();
    relative_paths_are_resolved_against_the_data_directory();
    absolute_paths_pass_through_unchanged();
    file_and_directory_checks();

    return mira_test::finish("paths");
}
