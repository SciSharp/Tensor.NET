file(GLOB_RECURSE SOURCES ./*.cpp)

message(${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

if(NOT WIN32 AND NN_CROSS_COMPILE)
    add_library(numnet.dll SHARED ${SOURCES})
    target_link_libraries(numnet.dll PRIVATE numnet_core_op.dll numnet_core_base.dll)
    set_target_properties(numnet.dll PROPERTIES
        SUFFIX ""
        LINK_FLAGS "-Wl,--add-stdcall-alias"
    )
    target_include_directories(numnet.dll PRIVATE ${PROJECT_SOURCE_DIR})
else()
    add_library(numnet SHARED ${SOURCES})
    set_target_properties(numnet PROPERTIES
        PREFFIX ""
    )
    target_link_libraries(numnet PRIVATE numnet_core_op numnet_core_base)
    target_include_directories(numnet PRIVATE ${PROJECT_SOURCE_DIR})
endif()