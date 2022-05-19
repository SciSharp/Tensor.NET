file(GLOB_RECURSE SOURCES ./*.cpp)

message(${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

if(NOT WIN32 AND NN_CROSS_COMPILE)
    add_library(tensornet.dll SHARED ${SOURCES})
    target_link_libraries(tensornet.dll PRIVATE tensornet_core_op.lib tensornet_core_base.lib)
    set_target_properties(tensornet.dll PROPERTIES
        SUFFIX ""
        LINK_FLAGS "-Wl,--add-stdcall-alias"
    )
    target_include_directories(tensornet.dll PRIVATE ${PROJECT_SOURCE_DIR})
else()
    add_library(tensornet SHARED ${SOURCES})
    set_target_properties(tensornet PROPERTIES
        PREFFIX ""
    )
    target_link_libraries(tensornet PRIVATE tensornet_core_op tensornet_core_base)
    target_include_directories(tensornet PRIVATE ${PROJECT_SOURCE_DIR})
endif()