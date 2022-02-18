
file(GLOB_RECURSE SOURCES common/*.cpp naive/*.cpp)

# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps")

enable_testing()

add_executable(
  core_test
  ${SOURCES}
)

target_link_libraries(
core_test
PUBLIC core_base
PUBLIC core_op
PRIVATE gtest_main
)

target_include_directories(core_test PUBLIC ${PROJECT_SOURCE_DIR})

include(GoogleTest)
gtest_discover_tests(core_test)
  