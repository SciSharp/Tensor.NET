cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-attributes -Wno-unused -Wno-parentheses")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
# set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

project(NumNet VERSION 1.0.0)

option(NN_WITH_TEST "Enable test for Num.NET." OFF)
option(NN_BUILD_TYPE "The build type for cmake, which includes Debug, Release, RelWithDebInfo" "Release")

set(CMAKE_BUILD_TYPE NN_BUILD_TYPE)

# link_libraries(NumNet PRIVATE core_base)
add_subdirectory(./core/base)
add_subdirectory(./core/test)
add_subdirectory(./core/op)
add_subdirectory(./apis)
add_subdirectory(./third_party/googletest)