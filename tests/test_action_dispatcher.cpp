#include "executor/action_dispatcher.hpp"
#include "executor/actions/open_application_action.hpp"
#include "appmgr/appmgr.hpp"
#include "resolver/app_resolver.hpp"
#include "resolver/application_config.hpp"

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>

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

const char* kTestAppsJson = R"({
    "version": 1,
    "applications": [
        {
            "name": "TrueApp",
            "aliases": ["truth", "verify-true"],
            "launch": {
                "method": "command",
                "command": "true"
            }
        },
        {
            "name": "SleepApp",
            "aliases": ["nap"],
            "launch": {
                "method": "command",
                "command": "sleep 0.1"
            }
        }
    ]
})";

class MockCustomAction : public mira::IAction
{
public:
    mira::Intent handled_intent() const override { return mira::Intent::Conversation; }
    mira::Result execute(const mira::ActionRequest& req) override
    {
        executed = true;
        last_request = req;
        return mira::Result::ok("conversation handled");
    }

    bool executed = false;
    mira::ActionRequest last_request;
};

void test_dispatch_unknown_intent()
{
    mira::ActionDispatcher dispatcher;

    mira::ActionRequest req;
    req.intent = mira::Intent::Unknown;
    req.original_text = "asdf random gibberish";

    mira::Result res = dispatcher.dispatch(req);
    TEST_ASSERT(!res.is_ok());
    TEST_ASSERT(res.status() == mira::ResultStatus::Rejected);
    TEST_ASSERT(res.message() == "unrecognized intent");
}

void test_dispatch_unregistered_intent()
{
    mira::ActionDispatcher dispatcher;

    mira::ActionRequest req;
    req.intent = mira::Intent::SearchWeb;
    req.original_text = "search for rust vs c++";

    mira::Result res = dispatcher.dispatch(req);
    TEST_ASSERT(!res.is_ok());
    TEST_ASSERT(res.status() == mira::ResultStatus::Unavailable);
    TEST_ASSERT(res.message() == "intent not supported yet");
}

void test_dispatch_custom_action()
{
    mira::ActionDispatcher dispatcher;

    auto action = std::make_unique<MockCustomAction>();
    MockCustomAction* raw_ptr = action.get();
    dispatcher.register_action(std::move(action));

    TEST_ASSERT(dispatcher.has_handler(mira::Intent::Conversation));
    TEST_ASSERT(!dispatcher.has_handler(mira::Intent::SearchWeb));

    mira::ActionRequest req;
    req.intent = mira::Intent::Conversation;
    req.original_text = "hello mira";

    mira::Result res = dispatcher.dispatch(req);
    TEST_ASSERT(res.is_ok());
    TEST_ASSERT(res.message() == "conversation handled");
    TEST_ASSERT(raw_ptr->executed);
    TEST_ASSERT(raw_ptr->last_request.original_text == "hello mira");
}

void test_dispatch_open_application_success()
{
    mira::ApplicationConfig cfg;
    mira::Result cfg_res = mira::ApplicationConfig::load_from_string(kTestAppsJson, cfg);
    TEST_ASSERT(cfg_res.is_ok());

    mira::AppResolver resolver(cfg);
    mira::AppManager app_mgr;
    mira::ActionDispatcher dispatcher;

    dispatcher.register_action(std::make_unique<mira::OpenApplicationAction>(resolver, app_mgr));

    mira::ActionRequest req;
    req.intent = mira::Intent::OpenApplication;
    req.tokens.valid = true;
    req.tokens.status = mira::TokenStatus::Success;
    req.tokens.type = mira::TokenType::TARGET;
    req.tokens.value = "TrueApp";
    req.original_text = "open TrueApp";

    mira::Result res = dispatcher.dispatch(req);
    TEST_ASSERT(res.is_ok());

    std::vector<pid_t> pids = app_mgr.running_pids();
    TEST_ASSERT(!pids.empty());

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    app_mgr.reap_children();
}

void test_dispatch_open_application_unconfigured()
{
    mira::ApplicationConfig cfg;
    mira::Result cfg_res = mira::ApplicationConfig::load_from_string(kTestAppsJson, cfg);
    TEST_ASSERT(cfg_res.is_ok());

    mira::AppResolver resolver(cfg);
    mira::AppManager app_mgr;
    mira::ActionDispatcher dispatcher;

    dispatcher.register_action(std::make_unique<mira::OpenApplicationAction>(resolver, app_mgr));

    mira::ActionRequest req;
    req.intent = mira::Intent::OpenApplication;
    req.tokens.valid = true;
    req.tokens.status = mira::TokenStatus::Success;
    req.tokens.type = mira::TokenType::TARGET;
    req.tokens.value = "non_existent_application";
    req.original_text = "open non_existent_application";

    mira::Result res = dispatcher.dispatch(req);
    TEST_ASSERT(!res.is_ok());
    TEST_ASSERT(res.status() == mira::ResultStatus::Rejected);
    TEST_ASSERT(res.message() == "not configured");
}

void test_dispatch_open_application_invalid_tokens()
{
    mira::ApplicationConfig cfg;
    mira::Result cfg_res = mira::ApplicationConfig::load_from_string(kTestAppsJson, cfg);
    TEST_ASSERT(cfg_res.is_ok());

    mira::AppResolver resolver(cfg);
    mira::AppManager app_mgr;
    mira::ActionDispatcher dispatcher;

    dispatcher.register_action(std::make_unique<mira::OpenApplicationAction>(resolver, app_mgr));

    mira::ActionRequest req;
    req.intent = mira::Intent::OpenApplication;
    req.tokens.valid = false;
    req.tokens.status = mira::TokenStatus::MissingTarget;
    req.tokens.value = "";
    req.original_text = "open";

    mira::Result res = dispatcher.dispatch(req);
    TEST_ASSERT(!res.is_ok());
    TEST_ASSERT(res.status() == mira::ResultStatus::Rejected);
    TEST_ASSERT(res.message() == "missing or invalid application target");
}

} // namespace

int main()
{
    test_dispatch_unknown_intent();
    test_dispatch_unregistered_intent();
    test_dispatch_custom_action();
    test_dispatch_open_application_success();
    test_dispatch_open_application_unconfigured();
    test_dispatch_open_application_invalid_tokens();

    std::cout << "All action_dispatcher tests passed successfully." << std::endl;
    return 0;
}

