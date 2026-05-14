#ifndef OPENVINO_ARMOR_H
#define OPENVINO_ARMOR_H

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>

struct DetectedArmor {
    std::vector<cv::Point2f> pnp_corners;   // 4 个角点（按你原来顺序）
    int number;                             // 数字类别 0~8
    float class_confidence;
    float confidence;
    cv::RotatedRect rect;
    float dist_to_center;
};

class OpenvinoArmorDetector {
public:
    // 直接传入 ONNX 模型路径 + 设备（CPU / GPU...）
    OpenvinoArmorDetector(const std::string& onnx_path,
                          const std::string& device = "CPU",
                          float conf_thresh = 0.65f,
                          float nms_thresh = 0.45f,
                          int detect_color = 0);   // 0:蓝 1:红

    // 执行检测
    std::vector<DetectedArmor> detect(const cv::Mat& img);

private:
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    ov::Tensor input_tensor;

    const int INPUT_WIDTH  = 640;
    const int INPUT_HEIGHT = 640;
    float conf_threshold_;
    float nms_threshold_;
    int detect_color_;

    cv::Mat preprocess(const cv::Mat& img);
    std::vector<DetectedArmor> postprocess(const float* output_data,
                                           const ov::Shape& output_shape,
                                           int img_w, int img_h);
};

#endif