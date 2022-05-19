file(GLOB_RECURSE SOURCES naive/*.cpp common/*.cpp)

if(NOT WIN32 AND NN_CROSS_COMPILE)
    add_library(tensornet_core_op.lib STATIC ${SOURCES})
    target_link_libraries(tensornet_core_op.lib PRIVATE tensornet_core_base.lib)
    set_target_properties(tensornet_core_op.lib PROPERTIES
        SUFFIX ""
        LINK_FLAGS "-Wl,--add-stdcall-alias"
    )
    target_include_directories(tensornet_core_op.lib PRIVATE ${PROJECT_SOURCE_DIR})
else()
    add_library(tensornet_core_op STATIC ${SOURCES})
    target_link_libraries(tensornet_core_op PRIVATE tensornet_core_base)
    target_include_directories(tensornet_core_op PRIVATE ${PROJECT_SOURCE_DIR})
endif()
