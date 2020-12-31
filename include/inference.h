/******************************************************************************/
/*!
File name: inference.h

Description:
This file define class of inference to load and run pb files.

Version: 0.1
Create date: 2020.12.11
Author: Chen Wei
Email: wei.chen@imotion.ai

Copyright (c) iMotion Automotive Technology (Suzhou) Co. Ltd. All rights reserved,
also regarding any disposal, exploitation, reproduction, editing, distribution,
as well as in the event of applications for industrial property rights.
*/
/******************************************************************************/

#ifndef TF_OD_C_INFERENCE_H
#define TF_OD_C_INFERENCE_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <ros/ros.h>
#include <ros/package.h>
#include <pcl_ros/point_cloud.h>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

#include "preprocess.h"
#include "postprocess.h"
#include "type.h"

class Inference {
public:
    Inference() = default;

    Inference(ros::NodeHandle node, std::string model_path, std::string input_op_name, std::vector<std::string> output_op_name);

    ~Inference();

    void LidarDataCallback(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr);

    void LidarPCD(std::string path);

    void RecogniseObject(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr, std::vector<BBox>& bbox);

    void LoadPb();

    void ExecutePb(cv::Mat input_data, std::vector<tensorflow::Tensor> &outputs);

    void CVMat2Tensor(cv::Mat input_data, tensorflow::Tensor *input_tensor);
private:
    std::string model_path_;
    std::string input_op_name_;
    std::vector<std::string> output_op_name_;
    tensorflow::Session *session_;

    std::shared_ptr<PreProcess> pre_process_;
    std::shared_ptr<PostProcess> post_process_;

    ros::Subscriber lidar_subscriber_;
};


#endif //TF_OD_C_INFERENCE_H
