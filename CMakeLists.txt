cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Debug")
# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps")
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

project(NumNet VERSION 0.1.0)

# link_libraries(NumNet PRIVATE core_base)
add_subdirectory(./core/base)
add_subdirectory(./core/test)
add_subdirectory(./core/op)
add_subdirectory(./apis)
add_subdirectory(./third_party/googletest)