# MiraV2 test suite.
#
# Deliberately dependency-free: no external test framework is used (ROADMAP
# principle: do not add frameworks that are not needed). Each test is a small
# executable that performs its checks and calls mira_test::finish().
#
# Tests must be runnable without a microphone, a display or a model file, and are
# linked against mira_core so they exercise exactly the code the executable links.

# Registers one test executable: mira_add_test(<ctest name> <sources...>)
function(mira_add_test name)
    add_executable(test_${name} ${ARGN})
    target_link_libraries(test_${name} PRIVATE mira_core)
    target_include_directories(test_${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(test_${name} PRIVATE ${MIRA_WARNING_FLAGS})
    add_test(NAME ${name} COMMAND test_${name})
endfunction()

# Phase 0: proves that CMake, the compiler, CTest and the library link work.
mira_add_test(smoke test_smoke.cpp)

# Phase 1: core interfaces.
mira_add_test(intent test_intent.cpp)
mira_add_test(result test_result.cpp)
mira_add_test(text_utils test_text_utils.cpp)
mira_add_test(paths test_paths.cpp)

# Tokenizer behaviour, including the cases that are deliberately unchanged from
# the pre-Phase-1 implementation.
mira_add_test(tokenizer test_tokenizer.cpp)

# Characterisation test for the legacy first-word splitter (cmdmgr/). It pins
# today's behaviour so the Phase 1 refactor and the Phase 8 removal cannot change
# it silently. cmdmgr.cpp is compiled here on purpose: it is not in mira_core.
mira_add_test(cmdmgr_baseline
    test_cmdmgr_baseline.cpp
    ${CMAKE_SOURCE_DIR}/cmdmgr/cmdmgr.cpp
)

# Phase 3: Application configuration schema and resolver.
mira_add_test(app_config test_app_config.cpp)
mira_add_test(app_resolver test_app_resolver.cpp)
# Phase 4: Application manager.
mira_add_test(appmgr test_appmgr.cpp)



