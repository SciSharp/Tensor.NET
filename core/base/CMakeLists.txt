set(CORE_BASE_SRC ./impl)
file(GLOB_RECURSE SOURCES ${CORE_BASE_SRC}/*.cpp)

if(NOT WIN32 AND NN_CROSS_COMPILE)
    add_library(numnet_core_base.lib STATIC ${SOURCES})
    set_target_properties(numnet_core_base.lib PROPERTIES
        SUFFIX ""
        LINK_FLAGS "-Wl,--add-stdcall-alias"
    )
    target_include_directories(numnet_core_base.lib PRIVATE ${PROJECT_SOURCE_DIR})
else()
    add_library(numnet_core_base STATIC ${SOURCES})
    target_include_directories(numnet_core_base PRIVATE ${PROJECT_SOURCE_DIR})
endif()

