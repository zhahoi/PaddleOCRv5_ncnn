// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef PPOCRV5_H
#define PPOCRV5_H

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <benchmark.h>
#include <net.h>
#include "cpu.h"

#include "ocr_dictionary.h"

struct Character
{
    int id;
    float prob;
};

struct Object
{
    cv::RotatedRect rrect;
    int orientation;
    float prob;
    std::vector<Character> text;
};

class PPOCRv5
{
public:
    PPOCRv5();
    ~PPOCRv5();

    int load(const char* det_parampath, const char* det_modelpath, const char* rec_parampath, const char* rec_modelpath, const std::string dict_path, bool use_fp16 = false, bool use_gpu = false);

    void set_target_size(int target_size);

    int detect(const cv::Mat& rgb, std::vector<Object>& objects);

    int recognize(const cv::Mat& rgb, Object& object);

    int detect_and_recognize(const cv::Mat& rgb, std::vector<Object>& objects);
    int draw(cv::Mat& rgb, const std::vector<Object>& objects);

    int draw_unsupported(cv::Mat& rgb);
    int draw_fps(cv::Mat& rgb);

protected:
    ncnn::Net ppocrv5_det;
    ncnn::Net ppocrv5_rec;
    OCRDictionary ocr_dictionary;
    int target_size;
    int character_dict_size;
};

#endif // PPOCRV5_H
