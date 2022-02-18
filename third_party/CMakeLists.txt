include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/3ea587050da9447536d0b55fece0a240273d9927.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)