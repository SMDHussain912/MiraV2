# MiraV2 test suite.
#
# Deliberately dependency-free: no external test framework is used yet
# (ROADMAP.md principle: do not add frameworks that are not needed). Each test
# is a small executable that performs its checks and calls mira_test::finish().
#
# Tests must be runnable without a microphone, a display or a model file.

# Trivial wiring test: proves CMake, the compiler and CTest are connected.
add_executable(test_smoke
    test_smoke.cpp
)
target_include_directories(test_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(test_smoke PRIVATE ${MIRA_WARNING_FLAGS})
add_test(NAME smoke COMMAND test_smoke)

# Characterisation test for the legacy command splitter (cmdmgr/).
# It pins today's behaviour so that the Phase 1 core refactor and the Phase 8
# retirement of CommandManager cannot change it silently.
add_executable(test_cmdmgr_baseline
    test_cmdmgr_baseline.cpp
    ${CMAKE_SOURCE_DIR}/cmdmgr/cmdmgr.cpp
)
target_include_directories(test_cmdmgr_baseline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
)
target_compile_options(test_cmdmgr_baseline PRIVATE ${MIRA_WARNING_FLAGS})
add_test(NAME cmdmgr_baseline COMMAND test_cmdmgr_baseline)
