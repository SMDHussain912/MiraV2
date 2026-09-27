#pragma once

#include <string>

namespace mira
{

// Outcome of an operation, following the failure model in ROADMAP.md section 6.3.
//
// The distinction between the statuses matters for how Mira responds:
//   Ok          the operation succeeded and was confirmed
//   Rejected    the request was understood but refused (not configured, ambiguous,
//               no target). The user has to change something to succeed.
//   Unavailable a required capability is missing (model file, backend, permission).
//               Not the user's mistake and not a failure of the operation.
//   Failed      the operation was attempted and failed (for example exec() error).
//   Unverified  something happened, but it could not be confirmed. Must never be
//               reported to the user as a success.
enum class ResultStatus
{
    Ok,
    Rejected,
    Unavailable,
    Failed,
    Unverified
};

// A status plus two strings:
//   message() user-facing text, short enough to be spoken
//   detail()  developer-facing context for logs; may be empty
class Result
{
public:
    Result();
    Result(ResultStatus status, std::string message, std::string detail = {});

    static Result ok(std::string message = {}, std::string detail = {});
    static Result rejected(std::string message, std::string detail = {});
    static Result unavailable(std::string message, std::string detail = {});
    static Result failed(std::string message, std::string detail = {});
    static Result unverified(std::string message, std::string detail = {});

    ResultStatus status() const;
    bool is_ok() const;

    // True for every status other than Ok.
    bool is_error() const;

    // True when the outcome is not a failure: Ok or Unverified.
    bool is_success_like() const;

    const std::string& message() const;
    const std::string& detail() const;

    // Status name, optionally with the message: "rejected: not configured".
    std::string to_string() const;

private:
    ResultStatus status_;
    std::string message_;
    std::string detail_;
};

} // namespace mira
