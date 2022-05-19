cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-attributes -Wno-unused -Wno-parentheses")
set(CMAKE_CXX_FLAGS "-fPIC")
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
# set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

project(TensorNet VERSION 0.1.0)

option(NN_WITH_TEST "Enable test for Num.NET." OFF)
option(NN_CROSS_COMPILE "Cross compile to build. So far only compile from Linux targeting Windows is supported" OFF)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Set the cmake build type to RelWithDebInfo since no build type was specified.")
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

if(NOT WIN32)
    if(NN_CROSS_COMPILE)
        message(STATUS "Use cross compile mode on linux targeting Windows.")
        set(PROJECT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/Windows/${CMAKE_BUILD_TYPE})
        set(CMAKE_SYSTEM_NAME Windows)

        set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)
        set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)
        set(CMAKE_RC_COMPILER x86_64-w64-mingw32-windres)
        set(CMAKE_RANLIB x86_64-w64-mingw32-ranlib)
    else()
        message(STATUS "Use host compile mode on Linux.")
        set(PROJECT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}/${CMAKE_BUILD_TYPE})
    endif()
else()
    message(STATUS "Use host compile mode on Windows.")
    set(PROJECT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SYSTEM_NAME}/${CMAKE_BUILD_TYPE})
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
message(STATUS "The outputs will be wrote into " ${PROJECT_BINARY_DIR} ".")
set(CMAKE_OUTPUT)


add_subdirectory(./core/base)
add_subdirectory(./core/op)
add_subdirectory(./apis)

if(NN_WITH_TEST)
    add_subdirectory(./third_party/googletest)
    add_subdirectory(./core/test)
endif()