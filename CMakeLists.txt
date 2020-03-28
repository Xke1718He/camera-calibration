cmake_minimum_required(VERSION 3.15)
project(camera_calibration)

set(CMAKE_CXX_STANDARD 11)
set(OpenCV_DIR "/usr/local/share/OpenCV")
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif()
endif()

include_directories(
        include
)
add_executable(camera_calibration main.cpp)
target_link_libraries(camera_calibration ${OpenCV_LIBS})