project(JDA-test)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
  message("OPENMP FOUND")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if (CMAKE_COMPILER_IS_GNUCC)
  # enable C99
  set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -std=c99")
endif()

add_definitions(-DJDA_EXPORTS)
add_library(libjda SHARED jda.h jda.c)

add_executable(jda-test main.cpp)
target_link_libraries(jda-test libjda ${OpenCV_LIBS})
