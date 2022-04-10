file(GLOB_RECURSE SOURCES ./*.cpp)

add_library(numnet SHARED ${SOURCES})

target_link_libraries(numnet PRIVATE core_op)

target_include_directories(numnet PRIVATE ${PROJECT_SOURCE_DIR})