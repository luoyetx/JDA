project(JDA)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)

include(libjda.cmake)

file(GLOB SRC src/*.cpp)

add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} libjda ${OpenCV_LIBS})
