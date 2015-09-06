# JDA

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include(${CMAKE_CURRENT_LIST_DIR}/3rdparty/liblinear/liblinear.cmake)

include_directories(${CMAKE_CURRENT_LIST_DIR}/3rdparty
                    ${CMAKE_CURRENT_LIST_DIR}/include)
file(GLOB SRC ${CMAKE_CURRENT_LIST_DIR}/include/jda/*.hpp
              ${CMAKE_CURRENT_LIST_DIR}/src/jda/*.cpp)

add_library(libjda STATIC ${SRC})
target_link_libraries(libjda liblinear ${OpenCV_LIBS})
