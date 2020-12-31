
#include "postprocess.h"

PostProcess::PostProcess(float score_threshold, float nms_threshold, int top_k)
{
    score_threshold_ = score_threshold;
    nms_threshold_ = nms_threshold;
    top_k_ = top_k;
}

PostProcess::~PostProcess()
{

}

void PostProcess::process(std::vector<tensorflow::Tensor> outputs, std::vector<BBox> &bbox)
{
    // 模型预测的目标数量
    int num = outputs[0].dim_size(1);

    // 第一个输出 confidence [1, 361, 5]
    float *obj_data = outputs[0].flat<float>().data();
    cv::Mat obj_probs(outputs[0].dim_size(1) * outputs[0].dim_size(2), 1, CV_32FC(1), obj_data);

    // 第二个输出 class [1, 361, 5, 4]
    float *class_data = outputs[1].flat<float>().data();
    cv::Mat class_probs(outputs[1].dim_size(1) * outputs[1].dim_size(2), outputs[1].dim_size(3), CV_32FC(1), class_data);

    // 第三个输出 bboxes [1, 361, 5, 6]
    float *boxes_data = outputs[2].flat<float>().data();
    cv::Mat boxes_probs(outputs[2].dim_size(1) * outputs[2].dim_size(2), outputs[2].dim_size(3), CV_32FC(1), boxes_data);




}

std::vector<float> PostProcess::calcscore(cv::Mat obj_probs, cv::Mat class_probs)
{



}


int PostProcess::argmax(const std::vector<float>& scores)
{

    return -1;
}

std::vector<int> PostProcess::argsort(const std::vector<float>& scores)
{
    int len = scores.size();
    std::vector<int> idx(len, 0);
    for(int i = 0; i < len; i++)
    {
        idx[i] = i;
    }

    std::sort(idx.begin(), idx.end(), [&scores](int i1, int i2){return scores[i1] < scores[i2];});

    return idx;
}


void PostProcess::sort()
{




//    index = np.argsort(-scores)
//    classes = classes[index][:top_k]
//    scores = scores[index][:top_k]
//    bboxes = bboxes[index][:top_k]
//    return classes, scores, bboxes
}