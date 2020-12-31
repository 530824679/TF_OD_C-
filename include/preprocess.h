/******************************************************************************/
/*!
File name: preprocess.h

Description:
This file define class of preprocess to transform point cloud to bev.

Version: 0.1
Create date: 2020.12.11
Author: Chen Wei
Email: wei.chen@imotion.ai

Copyright (c) iMotion Automotive Technology (Suzhou) Co. Ltd. All rights reserved,
also regarding any disposal, exploitation, reproduction, editing, distribution,
as well as in the event of applications for industrial property rights.
*/
/******************************************************************************/

#ifndef TF_OD_C_PREPROCESS_H
#define TF_OD_C_PREPROCESS_H

#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>

class PreProcess {
public:
    PreProcess() = default;

    PreProcess(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max);

    ~PreProcess();

    float Normalize(float value, float min, float max);

    void Calibrate(pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr);

    void InvalidRemove(pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr);

    void FilterROI(pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr);

    void GenerateBev(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr, cv::Mat &image);

private:
    float x_min_;
    float x_max_;
    float y_min_;
    float y_max_;
    float z_min_;
    float z_max_;
    float voxel_size_;
};


#endif //TF_OD_C_PREPROCESS_H
