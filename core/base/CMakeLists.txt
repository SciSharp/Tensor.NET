
# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps")

set(CORE_BASE_SRC ./impl)
file(GLOB_RECURSE SOURCES ${CORE_BASE_SRC}/*.cpp)

# message(${PROJECT_SOURCE_DIR})

add_library(core_base SHARED ${SOURCES})
target_include_directories(core_base PUBLIC ${PROJECT_SOURCE_DIR})
