# ==============================================================================
# whisper.cpp source dependency
# ==============================================================================
#
# MiraV2 builds whisper.cpp from source inside its own build tree instead of
# linking prebuilt shared libraries. Reasons:
#
#   * the prebuilt libraries previously vendored under speech/whisper_lib/lib
#     carried a RUNPATH pointing at a build directory outside the project, so
#     they only resolved on the machine that produced them (and libwhisper's
#     RUNPATH is not transitive, so the ggml libraries came from that external
#     directory regardless of the linker search path);
#   * a source build works on any Linux environment with a compiler and does not
#     require patching or trusting binary artifacts.
#
# Pinned revision:
#   repository : https://github.com/ggml-org/whisper.cpp.git
#   commit     : d09f61a708f3487afa956ff578e60eae5e7a233c
#   describes  : v1.9.4-181-gd09f61a7
#
# Note on whisper.h: speech/whisper.hpp and speech/whisper.cpp include the public
# header by relative path from the snapshot in speech/whisper_lib/include/, so
# the Whisper wrapper source is unchanged. That snapshot must stay
# byte-identical to the pinned revision; the check at the end of this file turns
# future version drift into a configure error instead of a silent ABI mismatch.
#
# Offline builds (or reusing an existing checkout) skip the clone entirely:
#   cmake -S . -B build -DFETCHCONTENT_SOURCE_DIR_WHISPER=/path/to/whisper.cpp

# Keep fetched sources inside the build tree so the project directory stays clean.
# This must be set before FetchContent is included: the module only installs its
# own default when the variable is still undefined, and a normal variable set by
# the module would otherwise shadow a cache entry created afterwards.
set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/deps"
    CACHE PATH "Directory where fetched dependencies are unpacked")

include(FetchContent)

set(MIRA_WHISPER_REPOSITORY "https://github.com/ggml-org/whisper.cpp.git"
    CACHE STRING "Git repository used to fetch whisper.cpp")

set(MIRA_WHISPER_REVISION "d09f61a708f3487afa956ff578e60eae5e7a233c"
    CACHE STRING "Pinned whisper.cpp revision (commit hash)")

# Build only the libraries MiraV2 needs. Set in the cache before the dependency
# is added, so whisper.cpp's own option() declarations keep these values.
set(WHISPER_BUILD_TESTS    OFF CACHE BOOL "whisper: build tests"    FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "whisper: build examples" FORCE)
set(WHISPER_BUILD_SERVER   OFF CACHE BOOL "whisper: build server"   FORCE)

# Third-party warnings are not MiraV2's to fix; our own targets still compile
# with MIRA_WARNING_FLAGS.
set(WHISPER_ALL_WARNINGS OFF CACHE BOOL "whisper: enable all warnings" FORCE)

# Static libraries: whisper.cpp and ggml are linked into the miraV2 executable,
# so there is no runtime library search path to get wrong, nothing to install,
# and no risk of picking up a stale shared library from elsewhere on the system.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "build shared libraries" FORCE)

FetchContent_Declare(
    whisper
    GIT_REPOSITORY "${MIRA_WHISPER_REPOSITORY}"
    GIT_TAG        "${MIRA_WHISPER_REVISION}"
    GIT_SHALLOW    FALSE
    GIT_PROGRESS   TRUE
)

FetchContent_MakeAvailable(whisper)

if(NOT TARGET whisper)
    message(FATAL_ERROR
        "whisper.cpp did not define the expected 'whisper' target. "
        "Check that MIRA_WHISPER_REVISION still points at a supported revision.")
endif()

# ------------------------------------------------------------------------------
# Guard: the header snapshot the Whisper wrapper compiles against must match the
# revision that is actually built and linked.
# ------------------------------------------------------------------------------

set(_mira_vendored_whisper_header
    "${CMAKE_SOURCE_DIR}/speech/whisper_lib/include/whisper.h")
set(_mira_upstream_whisper_header
    "${whisper_SOURCE_DIR}/include/whisper.h")

if(EXISTS "${_mira_vendored_whisper_header}" AND EXISTS "${_mira_upstream_whisper_header}")

    file(SHA256 "${_mira_vendored_whisper_header}" _mira_vendored_hash)
    file(SHA256 "${_mira_upstream_whisper_header}" _mira_upstream_hash)

    if(NOT _mira_vendored_hash STREQUAL _mira_upstream_hash)
        message(FATAL_ERROR
            "speech/whisper_lib/include/whisper.h does not match the pinned "
            "whisper.cpp revision (${MIRA_WHISPER_REVISION}).\n"
            "Refresh the header snapshot under speech/whisper_lib/include/ from "
            "the pinned revision so the wrapper does not compile against a "
            "header that differs from the library it links.")
    endif()

endif()
