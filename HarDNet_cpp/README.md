# HarDNet_cpp

#### Thanks to
[HarDNet](https://github.com/PingoLH/Pytorch-HarDNet)

## Environment
 * Ubuntu 20.04.3 LTS
 * ROS noetic 1.15.14
 * GCC 10.3.0
 * CMake 3.22.2
 * LibTorch 1.9.1
 * CUDA 11.1
 * OpenCV 4.2.0
 
## Download Large Files
[Download](https://o365cbnu-my.sharepoint.com/:u:/g/personal/2019132001_cbnu_ac_kr/EQjpX7EyELZLsphyej7jbUYBI3rRHNbbkP65s5hLL8BTuw?e=pXwxeA)  
[Tree](./inference_ros/src/inference_hardnet/data)

## Package Folder Tree
HarDNet_cpp
 * [train](./train)
   * [build](./train/build)
   * [src](./train/src)
     * [glob](./train/src/glob)
     * [fullyHardNet](./train/src/fullyHardNet)
 * [inference_ros](./inference_ros)
     * [src](./inference_ros/src)
         * [inference_hardnet](./inference_ros/src/inference_hardnet)
             * [src](./inference_ros/src/inference_hardnet/src)
               * [fullyHardNet](./inference_ros/src/inference_hardnet/src/fullyHardNet)
             * [launch](./inference_ros/src/inference_hardnet/launch)
             * [data](./inference_ros/src/inference_hardnet/data)
