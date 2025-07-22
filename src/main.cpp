#include "ppocrv5.h"

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#define access _access
#define F_OK 0
#else
#include <unistd.h>
#include <dirent.h>
#endif

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>

// 实例化模型
std::unique_ptr<PPOCRv5> ppOCRv5(new PPOCRv5());

// 定义权重路径
std::string det_parampath;
std::string det_modelpath;
std::string rec_parampath;
std::string rec_modelpath;
std::string dict_path;

// 用于获取当前路径
std::string getCurrentDir() {
    char buffer[1024];
#ifdef _WIN32
    _getcwd(buffer, sizeof(buffer));
#else
    getcwd(buffer, sizeof(buffer));
#endif
    return std::string(buffer);
}

// 测试路径是否正确
std::string findWeightsPath() {
    std::string currentDir = getCurrentDir();
    std::string weightsDir = currentDir + "/weights";
    std::ifstream testFile(weightsDir + "/PP_OCRv5_mobile_det.ncnn.bin");
    if (!testFile.is_open()) {
        std::cerr << "weights/ directory or PP_OCRv5_mobile_det.ncnn.bin not found in: " << weightsDir << std::endl;
        return "";
    }
    return weightsDir;
}

bool hasImageExtension(const std::string& filename) {
    std::string ext = filename.substr(filename.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <mode> <path>" << std::endl;
        std::cerr << "Mode: single - Process a single image" << std::endl;
        std::cerr << "      folder - Process all images in a folder" << std::endl;
        return -1;
    }

    std::string mode = argv[1];
    std::string path = argv[2];

    std::string weightsPath = findWeightsPath();
    if (weightsPath.empty()) return -1;

    det_parampath = weightsPath + "/PP_OCRv5_mobile_det.ncnn.param";
    det_modelpath = weightsPath + "/PP_OCRv5_mobile_det.ncnn.bin";
    rec_parampath = weightsPath + "/PP_OCRv5_mobile_rec.ncnn.param";
    rec_modelpath = weightsPath + "/PP_OCRv5_mobile_rec.ncnn.bin";
    dict_path = weightsPath + "/zh_dict.txt";

    ppOCRv5->load(det_parampath.c_str(), det_modelpath.c_str(), rec_parampath.c_str(), rec_modelpath.c_str(), dict_path, false, false);

    std::filesystem::create_directory("output");

    if (mode == "single") {
        cv::Mat image = cv::imread(path);
        if (image.empty()) {
            std::cerr << "Could not open or find the image at " << path << std::endl;
            return -1;
        }

        std::vector<Object> objects;
        auto start = std::chrono::high_resolution_clock::now();

        ppOCRv5->detect_and_recognize(image, objects);

        if (!objects.empty()) {
            ppOCRv5->draw(image, objects);
        }
        else {
            ppOCRv5->draw_unsupported(image);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "Processing time: " << elapsed.count() << " ms" << std::endl;

        std::string output_path = "output/result.jpg";
        cv::imwrite(output_path, image);
        std::cout << "Saved result to " << output_path << std::endl;

        cv::imshow("PPOCRv5 - Image", image);
        cv::waitKey(0);
    }
    else if (mode == "folder") {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (!hasImageExtension(filename)) continue;

                cv::Mat image = cv::imread(entry.path().string());
                if (image.empty()) {
                    std::cerr << "[WARN] Could not open " << filename << std::endl;
                    continue;
                }

                std::vector<Object> objects;
                auto start = std::chrono::high_resolution_clock::now();

                ppOCRv5->detect_and_recognize(image, objects);

                if (!objects.empty()) {
                    ppOCRv5->draw(image, objects);
                }
                else {
                    ppOCRv5->draw_unsupported(image);
                }

                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> elapsed = end - start;

                std::string output_path = "output/" + filename;
                cv::imwrite(output_path, image);

                std::cout << "Processed: " << filename << " | time: " << elapsed.count() << " ms | saved: " << output_path << std::endl;

                cv::imshow("PPOCRv5 - Folder", image);
                if (cv::waitKey(0) == 'q') break;
            }
        }
    }
    else {
        std::cerr << "Invalid mode. Use 'single' or 'folder'." << std::endl;
        return -1;
    }

    std::cout << "Processing complete" << std::endl;
    return 0;
}
