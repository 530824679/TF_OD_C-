/******************************************************************************/
/*!
File name: postprocess.h

Description:
This file define class of PostProcess include nms, sort function.

Version: 0.1
Create date: 2020.12.11
Author: Chen Wei
Email: wei.chen@imotion.ai

Copyright (c) iMotion Automotive Technology (Suzhou) Co. Ltd. All rights reserved,
also regarding any disposal, exploitation, reproduction, editing, distribution,
as well as in the event of applications for industrial property rights.
*/
/******************************************************************************/

#ifndef TF_OD_C_POSTPROCESS_H
#define TF_OD_C_POSTPROCESS_H

#include <math.h>
#include <vector>
#include <algorithm>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

#include "type.h"

class PostProcess {
public:
    PostProcess() = default;

    PostProcess(float score_threshold_, float nms_threshold_, int top_k_);

    ~PostProcess();

    std::vector<float> calcscore(cv::Mat obj_probs, cv::Mat class_probs);

    int argmax(const std::vector<float>& scores);

    int argmin(const std::vector<float>& scores);

    std::vector<int> argsort(const std::vector<float>& scores);

    void sort();

    void process(std::vector<tensorflow::Tensor> outputs, std::vector<BBox> &bbox);

private:
    float score_threshold_;
    float nms_threshold_;
    int top_k_;
};


#endif //TF_OD_C_POSTPROCESS_H
