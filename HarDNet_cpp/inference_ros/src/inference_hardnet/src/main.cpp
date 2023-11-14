#include <iostream>
#include <fstream>

#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <cv_bridge/cv_bridge.h>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include "fullyHardNet/fullyHardNet.h"

int cudaId = 0;
torch::Device device = torch::Device(cv::format("cuda:%d", cudaId));

HarDNet model(5);

void callback(const sensor_msgs::CompressedImage::ConstPtr& msg);

int main(int argc, char** argv) {
    ros::init(argc, argv, "inference_hardnet");
    ros::NodeHandle nh("~");

    std::string imgTopic;
    std::string modelWeightName;
    nh.param<std::string>("imgTopic", imgTopic, "/dev/device0");
    nh.param<std::string>("modelWeightName", modelWeightName, "/home/minkiri/catkin_ws/src/HarDNet_cpp/inference_ros/src/inference_hardnet/data/hardnet_outdoor.pt");

    ros::Subscriber sub = nh.subscribe(imgTopic, 1, callback);

    model->to(device);
    torch::load(model, modelWeightName);

    cv::namedWindow("img", cv::WINDOW_NORMAL);

    ros::spin();
}

void callback(const sensor_msgs::CompressedImage::ConstPtr& msg) {
    std::chrono::system_clock::time_point startTick = std::chrono::system_clock::now();

    model->eval();
    torch::NoGradGuard noGrad;

    cv::Mat img = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
    cv::resize(img, img, cv::Size(640, 360));

    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);
    torch::Tensor r = torch::from_blob(channels[2].ptr(), {img.rows, img.cols}, torch::kUInt8).to(torch::kFloat32).div(255.0).subtract(0.485).div(0.229);
    torch::Tensor g = torch::from_blob(channels[1].ptr(), {img.rows, img.cols}, torch::kUInt8).to(torch::kFloat32).div(255.0).subtract(0.456).div(0.224);
    torch::Tensor b = torch::from_blob(channels[0].ptr(), {img.rows, img.cols}, torch::kUInt8).to(torch::kFloat32).div(255.0).subtract(0.406).div(0.225);
    torch::Tensor imgBatchTensor = torch::cat({r, g, b}).view({1, 3, img.rows, img.cols}).to(device);

    torch::Tensor segPredBatchTensor = model(imgBatchTensor);
    torch::Tensor segPredImgTensor = std::get<1>(segPredBatchTensor.max(1, true))[0][0].to(torch::kUInt8).to(torch::kCPU);
    cv::Mat segPredImg = cv::Mat(img.rows, img.cols, CV_8UC1, segPredImgTensor.data_ptr<uint8_t>());
    cv::Mat segPredColorImg = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
    for(int j = 0; j < img.rows; j++) {
        for(int i = 0; i < img.cols; i++) {
            int predValue = segPredImg.at<uint8_t>(j, i);
            cv::Scalar color;
            if(predValue == 0) { //other
                color = cv::Scalar(255, 0, 0);
            }
            else if(predValue == 1) { //free space
                color = cv::Scalar(0, 255, 0);
            }
            else if(predValue == 2) { //dynamic object
                color = cv::Scalar(0, 0, 255);
            }
            segPredColorImg.at<cv::Vec3b>(j, i)[0] = color[0];
            segPredColorImg.at<cv::Vec3b>(j, i)[1] = color[1];
            segPredColorImg.at<cv::Vec3b>(j, i)[2] = color[2];
        }
    }
    img += segPredColorImg;

    cv::imshow("img", img);
    int wk = cv::waitKey(1);
    if(wk == 27) {
        exit(1);
    }

    std::chrono::duration<double> processingTime = std::chrono::system_clock::now() - startTick;
    //std::cout << processingTime.count() << std::endl;
}

