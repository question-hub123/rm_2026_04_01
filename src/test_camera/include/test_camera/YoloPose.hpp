#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

// 定义检测到的装甲板结构体
struct ArmorObject {
    cv::Rect box;
    int class_id;
    float confidence;
    std::vector<cv::Point2f> kps; // 4个角点
};

class YoloPose {
public:
    YoloPose(const std::string& model_path, int num_classes = 8, float conf_thresh = 0.5, float nms_thresh = 0.4);
    ~YoloPose() = default;

    std::vector<ArmorObject> detect(cv::Mat& src);
    void draw(cv::Mat& img, const std::vector<ArmorObject>& armors);

private:
    int num_classes_;
    float conf_thresh_;
    float nms_thresh_;
    cv::Size input_size_;

    // ONNXRuntime 核心组件
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;

    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;

    cv::Mat letterbox(const cv::Mat& source, cv::Size target_size, float& ratio, int& pad_w, int& pad_h);
};