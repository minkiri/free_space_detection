cmake_minimum_required(VERSION 3.0.2)
project(inference_hardnet)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  pcl_ros
  roscpp
  sensor_msgs
)
set(Torch_DIR "/home/minkiri/libtorch/share/cmake/Torch")


set(CMAKE_PREFIX_PATH "/home/minkiri/libtorch")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(INCLUDE_DIRS include CATKIN_DEPENDS  cv_bridge pcl_ros roscpp sensor_msgs)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  include
)

add_executable(${PROJECT_NAME} src/main.cpp src/fullyHardNet.cpp)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${TORCH_LIBRARIES}
  ${OpenCV_LIBS}
)
