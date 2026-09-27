// Unit tests for the operation result type (core/result.hpp).
//
// The statuses encode how Mira must respond: an "unverified" outcome has to be
// distinguishable from both success and failure, because reporting an unverified
// action as a success is not acceptable (ROADMAP.md section 6.3).

#include "core/result.hpp"
#include "test_util.hpp"

#include <string>

namespace
{

using mira::Result;
using mira::ResultStatus;

void a_default_result_is_ok()
{
    const Result result;

    MIRA_CHECK(result.is_ok());
    MIRA_CHECK(!result.is_error());
    MIRA_CHECK(result.is_success_like());

    MIRA_CHECK_EQ(result.status(), ResultStatus::Ok);
    MIRA_CHECK_EQ(result.message(), std::string(""));
    MIRA_CHECK_EQ(result.detail(), std::string(""));
}

void ok_keeps_message_and_detail()
{
    const Result result = Result::ok("Opening Firefox", "resolver: alias match");

    MIRA_CHECK_EQ(result.status(), ResultStatus::Ok);
    MIRA_CHECK(result.is_ok());
    MIRA_CHECK(!result.is_error());
    MIRA_CHECK(result.is_success_like());

    MIRA_CHECK_EQ(result.message(), std::string("Opening Firefox"));
    MIRA_CHECK_EQ(result.detail(), std::string("resolver: alias match"));
}

void rejected_is_an_error_the_user_can_fix()
{
    const Result result = Result::rejected(
        "Firefox is not configured",
        "resolver: no alias match for \"firefox\""
    );

    MIRA_CHECK_EQ(result.status(), ResultStatus::Rejected);
    MIRA_CHECK(result.is_error());
    MIRA_CHECK(!result.is_ok());
    MIRA_CHECK(!result.is_success_like());

    MIRA_CHECK_EQ(result.message(), std::string("Firefox is not configured"));
}

void unavailable_is_not_a_failure_of_the_operation()
{
    const Result result = Result::unavailable("I cannot capture the screen");

    MIRA_CHECK_EQ(result.status(), ResultStatus::Unavailable);
    MIRA_CHECK(result.is_error());
    MIRA_CHECK(!result.is_success_like());
}

void failed_reports_the_attempt()
{
    const Result result = Result::failed("I could not start Firefox", "execve: ENOENT");

    MIRA_CHECK_EQ(result.status(), ResultStatus::Failed);
    MIRA_CHECK(result.is_error());
    MIRA_CHECK(!result.is_success_like());

    MIRA_CHECK_EQ(result.detail(), std::string("execve: ENOENT"));
}

void unverified_is_neither_success_nor_failure()
{
    const Result result = Result::unverified("I could not confirm it worked");

    MIRA_CHECK_EQ(result.status(), ResultStatus::Unverified);

    // Not Ok, so it is an error in the sense of "do not treat as confirmed"...
    MIRA_CHECK(result.is_error());
    MIRA_CHECK(!result.is_ok());

    // ...but it must not be reported to the user as a failed operation either.
    MIRA_CHECK(result.is_success_like());
}

void to_string_contains_status_and_message()
{
    MIRA_CHECK_EQ(Result::ok().to_string(), std::string("ok"));
    MIRA_CHECK_EQ(Result::ok("done").to_string(), std::string("ok: done"));
    MIRA_CHECK_EQ(
        Result::rejected("not configured").to_string(),
        std::string("rejected: not configured")
    );
    MIRA_CHECK_EQ(
        Result::unavailable("no screen backend").to_string(),
        std::string("unavailable: no screen backend")
    );
    MIRA_CHECK_EQ(
        Result::failed("exec failed").to_string(),
        std::string("failed: exec failed")
    );
    MIRA_CHECK_EQ(
        Result::unverified("could not confirm").to_string(),
        std::string("unverified: could not confirm")
    );
}

} // namespace

int main()
{
    a_default_result_is_ok();
    ok_keeps_message_and_detail();
    rejected_is_an_error_the_user_can_fix();
    unavailable_is_not_a_failure_of_the_operation();
    failed_reports_the_attempt();
    unverified_is_neither_success_nor_failure();
    to_string_contains_status_and_message();

    return mira_test::finish("result");
}
