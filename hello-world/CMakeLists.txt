cmake_minimum_required(VERSION 3.5)

project(hello-world LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(hello-world main.cpp)

if (MINGW)
    message("Complier: MINGW")
elseif (MSVC)
    message("Complier: MSVC")
endif ()

if (WIN32)
    message("Platform: WIN32")
elseif (UNIX)
    message("Platform: UNIX")
else ()
    message("Platform: LINUX")
endif()
