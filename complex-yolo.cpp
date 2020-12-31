#include <string>
#include <vector>

// ros include
#include <ros/ros.h>
#include <ros/package.h>
#include "inference.h"


int main(int argc, char **argv)
{
    std::string model_path = "/home/chenwei/HDD/Project/TF_OD_C_Plus_Plus/model/frozen_model.pb";
    std::string input_op_name = "inputs";
    std::vector<std::string> output_op_name;
    output_op_name.push_back("reorg_layer/obj_probs");
    output_op_name.push_back("reorg_layer/class_probs");
    output_op_name.push_back("reorg_layer/bboxes_probs");

    try
    {
        ros::init(argc, argv, "TF_OD_NODE");
        ros::NodeHandle nh;
        Inference inf(nh, model_path, input_op_name, output_op_name);
        ROS_INFO("[%s]: Start TF OD ROS loop.\n", __func__);

        ros::spin();
    }
    catch (std::exception& e)
    {
        ROS_ERROR("[%s]: EXCEPTION '[%d]'.\n", __func__, e.what());
        return 1;
    }
    catch (...)
    {
        ROS_ERROR("[%s]: caught non-std EXCEPTION.\n", __func__);
        return 1;
    }




    return 0;
}