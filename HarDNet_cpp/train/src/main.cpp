#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <libgen.h>
#include <sys/stat.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <torch/torch.h>
#include <torch/nn/parallel/data_parallel.h>
#include <torch/optim/schedulers/lr_scheduler.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <fullyHardNet/fullyHardNet.h>
#include <glob/glob.h>

class CustomDatasetSegmentation : public torch::data::datasets::Dataset<CustomDatasetSegmentation> {
public:
    CustomDatasetSegmentation(const std::vector<std::pair<std::string, std::string>>& data, int imgWidth, int imgHeight) : data(data) {
        this->imgWidth = imgWidth;
        this->imgHeight = imgHeight;
    }

    torch::data::Example<> get(size_t index) {
        cv::Mat img = cv::imread(data[index].first, cv::IMREAD_UNCHANGED);
        cv::Mat segId = cv::imread(data[index].second, cv::IMREAD_UNCHANGED);
        cv::resize(img, img, cv::Size(this->imgWidth, this->imgHeight));
        cv::resize(segId, segId, cv::Size(this->imgWidth, this->imgHeight));
        std::vector<cv::Mat> channels(3);
        cv::split(img, channels);
        torch::Tensor r = torch::from_blob(channels[2].ptr(), {imgHeight, imgWidth}, torch::kUInt8).to(torch::kFloat32).div(255.0).subtract(0.485).div(0.229);
        torch::Tensor g = torch::from_blob(channels[1].ptr(), {imgHeight, imgWidth}, torch::kUInt8).to(torch::kFloat32).div(255.0).subtract(0.456).div(0.224);
        torch::Tensor b = torch::from_blob(channels[0].ptr(), {imgHeight, imgWidth}, torch::kUInt8).to(torch::kFloat32).div(255.0).subtract(0.406).div(0.225);
        torch::Tensor imgTensor = torch::cat({r, g, b}).view({3, imgHeight, imgWidth}).pin_memory();
        torch::Tensor segIdTensor = torch::from_blob(segId.ptr(), {imgHeight, imgWidth}, torch::kUInt8).to(torch::kInt64).pin_memory();
        return {imgTensor, segIdTensor};
    }

    torch::optional<size_t> size() const {
        return data.size();
    }
private:
    std::vector<std::pair<std::string, std::string>> data;
    int imgWidth;
    int imgHeight;
};

int main(int argc, char** argv) {
    if(!torch::cuda::is_available()) {
        std::cerr << "check cuda!!!" << std::endl;
        exit(1);
    }

    torch::manual_seed(time(0));
    torch::cuda::manual_seed(time(0));
    torch::cuda::manual_seed_all(time(0));

    int imgWidth = 640;
    int imgHeight = 360;
    int batchSize = 400;
    int totalEpoch = 1000;
    int startEpoch = 1; //minimum 1
    bool specificCheckpointFlag = true;
    std::string specificCheckpointModelName;
    if(specificCheckpointFlag) {
        specificCheckpointModelName = "/home/chungbuk/dk/2023/train_hardnet_multi_db/build/CNBMOOs_20230227/01000+model.pt";
    }
    if(startEpoch < 1) {
        std::cerr << "check startEpoch!!!" << std::endl;
        exit(1);
    }
    int useThreads = 20; //for loading dataset
    int cudaId = 0;
    torch::Device device = torch::Device(cv::format("cuda:%d", cudaId));
    std::vector<std::string> imgNameFormatLet;
    std::vector<std::string> segIdNameFormatLet;

    imgNameFormatLet.push_back("/home/chungbuk/dk/db_seg/Cityscape/leftImg8bit_trainvaltest/leftImg8bit/train/*/*.png");
    segIdNameFormatLet.push_back("/home/chungbuk/dk/db_seg/Cityscape/gtFine_trainvaltest_dkConversionV2/gtFine/train/*/*_labelIds.png");

    imgNameFormatLet.push_back("/home/chungbuk/dk/db_seg/NIA_DATA/images_png_dk2/train/*.png");
    segIdNameFormatLet.push_back("/home/chungbuk/dk/db_seg/NIA_DATA/gt_dk2_dkConversionV2/train/*.png");

    imgNameFormatLet.push_back("/home/chungbuk/dk/db_seg/BDD/bdd100k/seg/images/train/*.jpg");
    segIdNameFormatLet.push_back("/home/chungbuk/dk/db_seg/BDD/bdd100k/seg/labels_dkConversionV2/train/*.png");

    imgNameFormatLet.push_back("/home/chungbuk/dk/db_seg/Mapillary/training/images/*.jpg");
    segIdNameFormatLet.push_back("/home/chungbuk/dk/db_seg/Mapillary/training/v2.0/labels_id_dkConversionV2/*.png");

    imgNameFormatLet.push_back("/home/chungbuk/dk/db_seg/Own/*/png/60front/*.png");
    segIdNameFormatLet.push_back("/home/chungbuk/dk/db_seg/Own/*/png/60front_vit_adapter_infer_id_dkConversionV2/*.png");

    imgNameFormatLet.push_back("/home/chungbuk/dk/db_seg/Own_S/*/png/60front/*.png");
    segIdNameFormatLet.push_back("/home/chungbuk/dk/db_seg/Own_S/*/png/60front_vit_adapter_infer_id_dkConversionV2/*.png");
    imgNameFormatLet.push_back("/home/chungbuk/dk/db_seg/Own_S/*/png/60rear/*.png");
    segIdNameFormatLet.push_back("/home/chungbuk/dk/db_seg/Own_S/*/png/60rear_vit_adapter_infer_id_dkConversionV2/*.png");

    std::vector<std::pair<std::string, std::string>> dataPairLet;
    for(int j = 0; j < imgNameFormatLet.size(); j++) {
        std::string imgNameFormat = imgNameFormatLet[j];
        std::string segIdNameFormat = segIdNameFormatLet[j];
        std::vector<std::filesystem::path> imgFilesystemPathLet = glob::glob(imgNameFormat);
        std::vector<std::filesystem::path> segIdFilesystemPathLet = glob::glob(segIdNameFormat);
        std::sort(imgFilesystemPathLet.begin(), imgFilesystemPathLet.end());
        std::sort(segIdFilesystemPathLet.begin(), segIdFilesystemPathLet.end());
        if(imgFilesystemPathLet.size() != segIdFilesystemPathLet.size()) {
            std::cerr << "check data pair at db" << j << "!!!" << std::endl;
            exit(1);
        }
        for(int i = 0; i < imgFilesystemPathLet.size(); i++) {
            dataPairLet.push_back(std::make_pair(imgFilesystemPathLet[i].string(), segIdFilesystemPathLet[i].string()));
        }
    }
    std::srand(time(0));
    std::random_shuffle(dataPairLet.begin(), dataPairLet.end());
    auto dataset = CustomDatasetSegmentation(dataPairLet, imgWidth, imgHeight).map(torch::data::transforms::Stack<>());
    int dataSize = dataset.size().value();
    auto dataLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset), torch::data::DataLoaderOptions().batch_size(batchSize).workers(useThreads));

    HarDNet model(3);
    model->to(device);
    for (const auto &pair : model->named_parameters()) {
        std::string key = pair.key();
        float maxVal = torch::max(pair.value()).item<float>();
        float minVal = torch::min(pair.value()).item<float>();
        std::cout << key + " | max value: " << maxVal << ", min value: " << minVal << std::endl;
    }
    
    // torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.001).momentum(0.5));
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));
    torch::nn::CrossEntropyLoss criterion;
    
    cv::Mat graph = cv::Mat(1000, totalEpoch, CV_8UC3, cv::Scalar(255, 255, 255));
    if(startEpoch != 1) {
        std::string startEpochStr = cv::format("%05d+", startEpoch - 1);
        graph = cv::imread(startEpochStr + "loss.png", cv::IMREAD_UNCHANGED);
        torch::load(model, startEpochStr + "model.pt");
        torch::load(optimizer, startEpochStr + "optimizer.pt");
    }
    if(specificCheckpointFlag) {
        torch::load(model, specificCheckpointModelName);
    }
    // cv::namedWindow("show", cv::WINDOW_NORMAL);
    for(int i = startEpoch; i <= totalEpoch; i++) {
        model->train();
        float runningLoss = 0;
        int epoch = i;
        int forIdx = 0;
        std::chrono::system_clock::time_point currTick;
        std::chrono::system_clock::time_point prevTick = std::chrono::system_clock::now();
        for(const auto& batch : *dataLoader) {
            torch::Tensor imgBatchTensor = batch.data.to(device, true);
            torch::Tensor segIdBatchTensor = batch.target.to(device, true);
            torch::Tensor segPredBatchTensor = model(imgBatchTensor);
            torch::Tensor lossTensor = criterion(segPredBatchTensor, segIdBatchTensor);
            float localLoss = lossTensor.item<float>();
            runningLoss += localLoss; // runningLoss/dataIdx can be used <-> localLoss

            optimizer.zero_grad();
            lossTensor.backward();
            optimizer.step();
            int dataIdx = std::min(dataSize, (forIdx + 1)*batchSize);
            forIdx++;

            torch::Tensor rTensor = imgBatchTensor[0][0].mul(0.229).add(0.485).mul(255.0).to(torch::kUInt8).to(torch::kCPU);
            torch::Tensor gTensor = imgBatchTensor[0][1].mul(0.224).add(0.456).mul(255.0).to(torch::kUInt8).to(torch::kCPU);
            torch::Tensor bTensor = imgBatchTensor[0][2].mul(0.225).add(0.406).mul(255.0).to(torch::kUInt8).to(torch::kCPU);
            std::vector<cv::Mat> bgr(3);
            bgr[0] = cv::Mat(imgHeight, imgWidth, CV_8UC1, bTensor.data_ptr<uint8_t>());
            bgr[1] = cv::Mat(imgHeight, imgWidth, CV_8UC1, gTensor.data_ptr<uint8_t>());
            bgr[2] = cv::Mat(imgHeight, imgWidth, CV_8UC1, rTensor.data_ptr<uint8_t>());
            cv::Mat img;
            cv::merge(bgr, img);
            torch::Tensor segPredImgTensor = std::get<1>(segPredBatchTensor.max(1, true))[0][0].to(torch::kUInt8).to(torch::kCPU);
            cv::Mat segPredImg = cv::Mat(imgHeight, imgWidth, CV_8UC1, segPredImgTensor.data_ptr<uint8_t>());
            cv::normalize(segPredImg, segPredImg, 0, 255, cv::NORM_MINMAX);

            cv::circle(graph, cv::Point(epoch, graph.rows - (int)(500*localLoss + 1)), 3, cv::Scalar(255, 0, 0), -1);

            // cv::Mat resizedImg, resizedSegPredImg;
            // float showRatio = ((float)graph.rows/2.0)/((float)img.rows);
            // cv::resize(img, resizedImg, cv::Size(showRatio*img.cols, graph.rows/2), CV_8UC3);
            // cv::resize(segPredImg, resizedSegPredImg, cv::Size(showRatio*segPredImg.cols, graph.rows/2), CV_8UC3, cv::INTER_NEAREST);
            // cv::cvtColor(resizedSegPredImg, resizedSegPredImg, cv::COLOR_GRAY2BGR);
            // cv::Mat show = cv::Mat(graph.rows, resizedImg.cols + graph.cols, CV_8UC3);
            // resizedImg.copyTo(show(cv::Rect(0, 0, resizedImg.cols, resizedImg.rows)));
            // resizedSegPredImg.copyTo(show(cv::Rect(0, resizedImg.rows, resizedSegPredImg.cols, resizedSegPredImg.rows)));
            // graph.copyTo(show(cv::Rect(resizedImg.cols, 0, graph.cols, graph.rows)));
            // cv::imshow("show", show);

            currTick = std::chrono::system_clock::now();
            std::chrono::duration<double> processingTime = currTick - prevTick;
            prevTick = currTick;
            size_t cudaFreeByteT;
            size_t cudaTotalByteT;
            cudaSetDevice(cudaId);
            cudaMemGetInfo(&cudaFreeByteT, &cudaTotalByteT);
            float cudaFree = (float)(cudaFreeByteT/1024.0)/1024.0;
            float cudaTotal = (float)(cudaTotalByteT/1024.0)/1024.0;
            float cudaUsed = cudaTotal - cudaFree;
            std::cout << "Epoch " << epoch << ": " << dataIdx << "/" << dataSize << " | Loss: " << localLoss << " | Processing Time: " << processingTime.count() << "s" << " | CUDA Memory Usage: " << cudaUsed << "MB/" << cudaTotal << "MB" << std::endl;

            if((forIdx == std::ceil((float)dataSize/(float)(batchSize))) && (epoch%10 == 0)) {
                std::string epochStr = cv::format("%05d+", epoch);
                torch::save(model, epochStr + "model.pt");
                torch::save(optimizer, epochStr + "optimizer.pt");
                cv::imwrite(epochStr + "loss.png", graph);
                cv::imwrite(epochStr + "valid_img.png", img);
                cv::imwrite(epochStr + "valid_seg_pred_img.png", segPredImg);
            }

            // int wk = cv::waitKey(1);
            // if(wk == 27) {
            //     std::string epochStr = cv::format("%05d+", epoch);
            //     torch::save(model, epochStr + "model_forced_stop.pt");
            //     torch::save(optimizer, epochStr + "optimizer_forced_stop.pt");
            //     cv::imwrite(epochStr + "loss_forced_stop.png", graph);
            //     cv::imwrite(epochStr + "valid_img_forced_stop.png", img);
            //     cv::imwrite(epochStr + "valid_seg_pred_img_forced_stop.png", segPredImg);
            //     exit(1);
            // }
        }
    }

    return 0;
}


