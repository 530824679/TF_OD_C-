#include "inference.h"

Inference::Inference(ros::NodeHandle node, std::string model_path, std::string input_op_name, std::vector<std::string> output_op_name)
{
    model_path_ = model_path;
    input_op_name_ = input_op_name;
    output_op_name_ = output_op_name;
    pre_process_ = std::make_shared<PreProcess>(0.0, 60.8, -30.4, 30.4, -3.0, 3.0);
    post_process_ = std::make_shared<PostProcess>(0.5, 0.5, 100);

    LoadPb();
    LidarPCD("/home/chenwei/HDD/Project/Complex-YOLOv2/test/008231.pcd");
    //lidar_subscriber_ = node.subscribe("/livox/lidar", 100, &Inference::LidarDataCallback, this, ros::TransportHints().reliable().tcpNoDelay(true));
}

Inference::~Inference()
{
    session_->Close();
}

void Inference::LidarDataCallback(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr)
{
    std::vector<BBox> bbox;
    RecogniseObject(in_cloud_ptr, bbox);
}

void Inference::LidarPCD(std::string path)
{
    // 创建点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr (new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(path, *in_cloud_ptr) == -1) {
        PCL_ERROR("PCD file reading failed.");
        return;
    }
    std::vector<BBox> bbox;
    RecogniseObject(in_cloud_ptr, bbox);
}

void Inference::RecogniseObject(const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr, std::vector<BBox> &bbox)
{
    // pre-process point cloud tranform bev
    cv::Mat bev_map = cv::Mat::zeros(608, 608, CV_8UC3);
    pre_process_->InvalidRemove(in_cloud_ptr);
    pre_process_->Calibrate(in_cloud_ptr);
    pre_process_->FilterROI(in_cloud_ptr);
    pre_process_->GenerateBev(in_cloud_ptr, bev_map);

    // execute pb
    std::vector<tensorflow::Tensor> outputs;
    ExecutePb(bev_map, outputs);

    // post process
    post_process_->process(outputs, bbox);
}

void Inference::LoadPb()
{
    // create session
    tensorflow::SessionOptions options;
    options.config.mutable_gpu_options()->set_allow_growth(true);
    options.config.mutable_gpu_options()->set_visible_device_list("0");

    tensorflow::Status status = tensorflow::NewSession(options, &session_);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        throw std::runtime_error("Could not create Tensorflow session.");
    }

    // read graph in the protobuf
    tensorflow::GraphDef graphDef;
    tensorflow::Status status_od_net = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), model_path_, &graphDef);
    if (!status_od_net.ok())
    {
        throw std::runtime_error("Error reading graph definition from " + model_path_ + ": " + status_od_net.ToString());
    }

    // add the graph to the session
    tensorflow::Status status_create = session_->Create(graphDef);
    if (!status_create.ok())
    {
        throw std::runtime_error("Error creating graph in session failed : " + status_create.ToString());
    }
}

void Inference::CVMat2Tensor(cv::Mat input_data, tensorflow::Tensor *input_tensor)
{
    // 创建一个指向tensor的内容的指针
    float *tensorDataPtr = input_tensor->flat<float>().data();

    //创建一个Mat，与tensor的指针绑定,改变这个Mat的值，就相当于改变tensor的值
    cv::Mat fake_mat(input_data.rows, input_data.cols, CV_32FC(input_data.channels()), tensorDataPtr);

    input_data.convertTo(fake_mat, CV_32FC(input_data.channels()));
}

void Inference::ExecutePb(cv::Mat input_data, std::vector<tensorflow::Tensor> &outputs)
{
    tensorflow::Tensor input_tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, {1, input_data.rows, input_data.cols, 3});
    CVMat2Tensor(input_data, &(input_tensor));
    std::cout << input_tensor.DebugString() << std::endl;

    tensorflow::Status status = session_->Run({{input_op_name_, input_tensor}}, {output_op_name_}, {}, &outputs);
    if (!status.ok())
    {
        throw std::runtime_error("Error running graph in session failed : " + status.ToString());
    }

    //把输出值给提取出
    std::cout << "Output tensor size:" << outputs.size() << std::endl;  //3
    for (int i = 0; i < outputs.size(); i++)
    {
        std::cout << outputs[i].DebugString() << std::endl;   // [1, 50], [1, 50], [1, 50, 4]
    }
}