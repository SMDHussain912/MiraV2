// Trivial wiring test.
//
// Purpose (ROADMAP.md Phase 0): prove that CMake, the compiler and CTest are
// connected and that the suite can be run without any hardware, model file or
// display. It is intentionally not a test of MiraV2 behaviour; real unit tests
// start with the core interfaces in Phase 1.

#include "test_util.hpp"

#include <string>

int main()
{
    MIRA_CHECK(1 + 1 == 2);
    MIRA_CHECK_EQ(std::string("mira"), std::string("mira"));
    MIRA_CHECK_EQ(sizeof(int), size_t{4});

    return mira_test::finish("smoke");
}
