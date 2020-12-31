
#include "preprocess.h"


PreProcess::PreProcess(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max)
{
    x_min_ = x_min;
    x_max_ = x_max;
    y_min_ = y_min;
    y_max_ = y_max;
    z_min_ = z_min;
    z_max_ = z_max;
    voxel_size_ = 0.1;
}

PreProcess::~PreProcess()
{

}

float PreProcess::Normalize(float value, float min, float max)
{
    return ((value - min) / float(max - min));
}

void PreProcess::Calibrate(pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr)
{
    // Calibration angle theta 弧度单位
    float roll = -0.0404126;
    float pitch = 0.170683;
    float yaw = -0.069972;
    if (roll)
    {
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.rotate(Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX()));
        pcl::transformPointCloud(*in_cloud_ptr, *in_cloud_ptr, transform);
    }
    if (pitch)
    {
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.rotate(Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()));
        pcl::transformPointCloud(*in_cloud_ptr, *in_cloud_ptr, transform);
    }
    if (yaw)
    {
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.rotate(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
        pcl::transformPointCloud(*in_cloud_ptr, *in_cloud_ptr, transform);
    }

}

void PreProcess::InvalidRemove(pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr)
{
    for (auto it = (*in_cloud_ptr).begin(); it != (*in_cloud_ptr).end();)
    {
        if (((*it).x == 0) && ((*it).y == 0) && ((*it).z == 0))
        {
            (*in_cloud_ptr).erase(it++);
        }
        else
        {
            it++;
        }
    }
}


void PreProcess::FilterROI(pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr)
{
    pcl::PassThrough<pcl::PointXYZI> cloudXFilter, cloudYFilter, cloudZFilter;
    cloudXFilter.setInputCloud(in_cloud_ptr);
    cloudXFilter.setFilterFieldName("x");
    cloudXFilter.setFilterLimits(x_min_, x_max_);
    cloudXFilter.filter(*in_cloud_ptr);

    cloudYFilter.setInputCloud(in_cloud_ptr);
    cloudYFilter.setFilterFieldName("y");
    cloudYFilter.setFilterLimits(y_min_, y_max_);
    cloudYFilter.filter(*in_cloud_ptr);

    cloudZFilter.setInputCloud(in_cloud_ptr);
    cloudZFilter.setFilterFieldName("z");
    cloudZFilter.setFilterLimits(z_min_, z_max_);
    cloudZFilter.filter(*in_cloud_ptr);
}


void PreProcess::GenerateBev(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr, cv::Mat &image)
{
    int x_max = ceil((y_max_ - y_min_) / voxel_size_);
    int y_max = ceil((x_max_ - x_min_) / voxel_size_);
    float bev_map[y_max][x_max][3] = {0};

    // 0:density, 1:height, 2:intensity;
    for (int k = 0; k < (*in_cloud_ptr).size(); k++) {
        float x = (*in_cloud_ptr)[k].x;
        float y = (*in_cloud_ptr)[k].y;
        float z = (*in_cloud_ptr)[k].z;
        float intensity = (*in_cloud_ptr)[k].intensity;

        int x_img = int(-y / voxel_size_) - floor(y_min_ / voxel_size_);
        int y_img = int(-x / voxel_size_) + floor(x_max_ / voxel_size_);
        float pixel_values = Normalize(z, z_min_, z_max_);

        if (pixel_values > bev_map[y_img][x_img][1])
            bev_map[y_img][x_img][1] = pixel_values;
        if (intensity > bev_map[y_img][x_img][2])
            bev_map[y_img][x_img][2] = intensity / 255.0;
        bev_map[y_img][x_img][0] += 1;
    }

    for (int j = 0; j < y_max; j++) {
        for (int i = 0; i < x_max; i++) {
            if (bev_map[j][i][0] > 0)
                bev_map[j][i][0] = std::min(1.0, log(bev_map[j][i][0] + 1) / log(64));
        }
    }

    for (int j = 0; j < y_max; j++) {
        for (int i = 0; i < x_max; i++) {
            image.at<cv::Vec3b>(j, i)[0] = bev_map[j][i][0];
            image.at<cv::Vec3b>(j, i)[1] = bev_map[j][i][1];
            image.at<cv::Vec3b>(j, i)[2] = bev_map[j][i][2];
            std::cout << bev_map[j][i][0] << ", " << bev_map[j][i][1] << ", " << bev_map[j][i][2] << std::endl;
        }
    }
}
