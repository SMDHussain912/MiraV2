#pragma once

// Minimal, dependency-free helpers for the MiraV2 test suite.
//
// A test is a plain executable: it performs checks with MIRA_CHECK /
// MIRA_CHECK_EQ and returns mira_test::finish("<suite name>"), which reports
// the result and yields a process exit code of 0 (pass) or 1 (failure).
//
// Failures are reported as they happen rather than aborting the run, so a
// single execution shows every failing case.

#include <iostream>
#include <sstream>
#include <string>

namespace mira_test
{

inline int& failure_count()
{
    static int count = 0;
    return count;
}

inline void report_failure(
    const char* file,
    int line,
    const std::string& message)
{
    ++failure_count();

    std::cerr
        << file << ':' << line
        << ": FAIL: " << message
        << '\n';
}

// Human-readable rendering of values in failure output.

inline std::string to_text(const std::string& value)
{
    return '"' + value + '"';
}

inline std::string to_text(const char* value)
{
    if (value == nullptr)
    {
        return "(null)";
    }

    return std::string("\"") + value + '"';
}

inline std::string to_text(bool value)
{
    return value ? "true" : "false";
}

template <typename T>
inline std::string to_text(const T& value)
{
    std::ostringstream out;
    out << value;
    return out.str();
}

inline int finish(const std::string& suite)
{
    if (failure_count() == 0)
    {
        std::cout << suite << ": all checks passed\n";
        return 0;
    }

    std::cerr
        << suite << ": "
        << failure_count() << " check(s) failed\n";

    return 1;
}

} // namespace mira_test

#define MIRA_CHECK(condition)                                                \
    do                                                                       \
    {                                                                        \
        if (!(condition))                                                    \
        {                                                                    \
            ::mira_test::report_failure(__FILE__, __LINE__, #condition);     \
        }                                                                    \
    } while (false)

#define MIRA_CHECK_EQ(actual, expected)                                      \
    do                                                                       \
    {                                                                        \
        const auto& mira_actual = (actual);                                  \
        const auto& mira_expected = (expected);                              \
                                                                             \
        if (!(mira_actual == mira_expected))                                 \
        {                                                                    \
            ::mira_test::report_failure(                                     \
                __FILE__,                                                    \
                __LINE__,                                                    \
                std::string(#actual) + " == " + #expected +                  \
                    " (got " + ::mira_test::to_text(mira_actual) +           \
                    ", expected " + ::mira_test::to_text(mira_expected) +    \
                    ")");                                                    \
        }                                                                    \
    } while (false)
