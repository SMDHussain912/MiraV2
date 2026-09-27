cmake_minimum_required(VERSION 3.16)

project(miraV2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------- build type
# Without this, a plain `cmake -S . -B build` produces an unoptimised binary
# with no debug info. Default to RelWithDebInfo unless the user chose a type.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
    message(STATUS "CMAKE_BUILD_TYPE not set; defaulting to RelWithDebInfo")
endif()

# ---------------------------------------------------------------- warnings
set(MIRA_WARNING_FLAGS -Wall -Wextra)

# Known baseline: audio/microphone.cpp produces 23 warnings with these flags
# (unused parameters, missing initializers in PipeWire event structs, member
# initialisation order). They are pre-existing and are scheduled to be fixed
# together with the microphone lifecycle work (ROADMAP.md Phase 14). New code
# must not add warnings.

# ---------------------------------------------------------------- dependencies
find_package(PkgConfig REQUIRED)
pkg_check_modules(PIPEWIRE REQUIRED libpipewire-0.3)

# whisper.cpp is built from source as part of this build; see cmake/whisper.cmake
# for the pinned revision and the available configuration knobs.
include(cmake/whisper.cmake)

# ---------------------------------------------------------------- mira_core
# MiraV2's own C++ library: the core interfaces and the intent-aware tokenizer.
# Kept separate from the executable so components can be unit tested without the
# audio, input or speech code, and so the executable and the tests link exactly
# the same code.
#
# The legacy cmdmgr/ splitter is deliberately not part of this library: it is
# compiled into the executable and its characterisation test only, and is removed
# in ROADMAP Phase 8.
add_library(mira_core STATIC
    core/action.cpp
    core/intent.cpp
    core/result.cpp
    core/text_utils.cpp
    core/paths.cpp
    tokenizer/tokenizer.cpp
)

target_include_directories(mira_core PUBLIC
    ${CMAKE_SOURCE_DIR}
)

# core/paths.cpp resolves development-time data paths through this definition, so
# files such as the Whisper model are found regardless of the working directory
# the binary is started from.
target_compile_definitions(mira_core PUBLIC
    MIRA_SOURCE_ROOT="${CMAKE_SOURCE_DIR}"
)

target_compile_options(mira_core PRIVATE ${MIRA_WARNING_FLAGS})

add_executable(miraV2
	main.cpp
	speech/whisper.cpp
	audio/microphone.cpp
	input/hotkey.cpp
	cmdmgr/cmdmgr.cpp
)

target_include_directories(miraV2 PRIVATE
    ${PIPEWIRE_INCLUDE_DIRS}
)

# No link directories: the whisper and ggml libraries are built in this tree as
# static libraries and linked through the `whisper` target below.
target_link_libraries(miraV2 PRIVATE
    mira_core
    whisper
    ${PIPEWIRE_LIBRARIES}
)

target_compile_options(miraV2 PRIVATE ${MIRA_WARNING_FLAGS})

# ---------------------------------------------------------------- tests
option(MIRA_BUILD_TESTS "Build the MiraV2 test suite" ON)

if(MIRA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
