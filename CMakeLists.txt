project(JDA)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include(3rdparty/liblinear/liblinear.cmake)

include_directories(3rdparty)
include_directories(include)
file(GLOB SRC include/jda/*.h*
              src/jda/*.c*)

add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} liblinear ${OpenCV_LIBS})