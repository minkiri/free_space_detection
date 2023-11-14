cmake_minimum_required(VERSION 3.0.2)
project(train_hardnet)

add_compile_options(-std=c++17)

set(CMAKE_PREFIX_PATH "/home/chungbuk/libtorch")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  ${TORCH_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  src
)

add_executable(${PROJECT_NAME} src/main.cpp src/fullyHardNet/fullyHardNet.cpp src/glob/glob.cpp)

target_link_libraries(${PROJECT_NAME}
  ${TORCH_LIBRARIES}
  ${OpenCV_LIBS}
)
