file(GLOB_RECURSE SOURCES naive/*.cpp common/*.cpp)

if(NOT WIN32 AND NN_CROSS_COMPILE)
    add_library(numnet_core_op.dll SHARED ${SOURCES})
    target_link_libraries(numnet_core_op.dll PRIVATE numnet_core_base.dll)
    set_target_properties(numnet_core_op.dll PROPERTIES
        SUFFIX ""
        LINK_FLAGS "-Wl,--add-stdcall-alias"
    )
    target_include_directories(numnet_core_op.dll PRIVATE ${PROJECT_SOURCE_DIR})
else()
    add_library(numnet_core_op SHARED ${SOURCES})
    target_link_libraries(numnet_core_op PRIVATE numnet_core_base)
    target_include_directories(numnet_core_op PRIVATE ${PROJECT_SOURCE_DIR})
endif()
