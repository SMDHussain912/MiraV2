#include "core/result.hpp"

namespace mira
{

namespace
{

const char* status_name(ResultStatus status)
{
    switch (status)
    {
        case ResultStatus::Ok:          return "ok";
        case ResultStatus::Rejected:    return "rejected";
        case ResultStatus::Unavailable: return "unavailable";
        case ResultStatus::Failed:      return "failed";
        case ResultStatus::Unverified:  return "unverified";
    }

    return "unknown";
}

} // namespace

Result::Result()
    : status_(ResultStatus::Ok),
      message_(),
      detail_()
{
}

Result::Result(
    ResultStatus status,
    std::string message,
    std::string detail)
    : status_(status),
      message_(std::move(message)),
      detail_(std::move(detail))
{
}

Result Result::ok(std::string message, std::string detail)
{
    return Result(ResultStatus::Ok, std::move(message), std::move(detail));
}

Result Result::rejected(std::string message, std::string detail)
{
    return Result(ResultStatus::Rejected, std::move(message), std::move(detail));
}

Result Result::unavailable(std::string message, std::string detail)
{
    return Result(ResultStatus::Unavailable, std::move(message), std::move(detail));
}

Result Result::failed(std::string message, std::string detail)
{
    return Result(ResultStatus::Failed, std::move(message), std::move(detail));
}

Result Result::unverified(std::string message, std::string detail)
{
    return Result(ResultStatus::Unverified, std::move(message), std::move(detail));
}

ResultStatus Result::status() const
{
    return status_;
}

bool Result::is_ok() const
{
    return status_ == ResultStatus::Ok;
}

bool Result::is_error() const
{
    return status_ != ResultStatus::Ok;
}

bool Result::is_success_like() const
{
    return status_ == ResultStatus::Ok || status_ == ResultStatus::Unverified;
}

const std::string& Result::message() const
{
    return message_;
}

const std::string& Result::detail() const
{
    return detail_;
}

std::string Result::to_string() const
{
    std::string text = status_name(status_);

    if (!message_.empty())
    {
        text += ": ";
        text += message_;
    }

    return text;
}

} // namespace mira
