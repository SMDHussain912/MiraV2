cmake_minimum_required(VERSION 3.16)

project(miraV2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PIPEWIRE REQUIRED libpipewire-0.3)

add_executable(miraV2
	main.cpp
	speech/whisper.cpp
	audio/microphone.cpp
	input/hotkey.cpp
	cmdmgr/cmdmgr.cpp
)

target_include_directories(miraV2 PRIVATE
    speech/whisper_lib/include
    ${PIPEWIRE_INCLUDE_DIRS}
)

target_link_directories(miraV2 PRIVATE
	speech/whisper_lib/lib
)

target_link_libraries(miraV2 PRIVATE
    whisper
    ${PIPEWIRE_LIBRARIES}
)
