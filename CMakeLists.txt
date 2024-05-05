cmake_minimum_required(VERSION 3.5)  # CMake version check
project(almanak)      

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()

set(CMAKE_PREFIX_PATH pytorch_libs/libtorch)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add executable target with source files listed in SOURCE_FILES variable
add_executable(${PROJECT_NAME} src/main.cpp src/simplest_fc_layer.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC include pytorch_libs)
target_link_libraries(${PROJECT_NAME} PRIVATE "${TORCH_LIBRARIES}")