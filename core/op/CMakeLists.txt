
file(GLOB_RECURSE SOURCES naive/*.cpp ./*.h ./*.cpp naive/*.h)

# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps")

message(${SOURCES})

add_library(core_op SHARED ${SOURCES})
target_link_libraries(core_op PUBLIC core_base)
target_include_directories(core_op PUBLIC ${PROJECT_SOURCE_DIR})
