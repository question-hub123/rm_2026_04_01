#include "YoloPose.hpp"
#include <opencv2/dnn.hpp> // 仅用来做预处理和NMS

YoloPose::YoloPose(const std::string& model_path, int num_classes, float conf_thresh, float nms_thresh)
    : num_classes_(num_classes), conf_thresh_(conf_thresh), nms_thresh_(nms_thresh), input_size_(640, 640),
      env_(ORT_LOGGING_LEVEL_WARNING, "YoloPose"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU))
{
    // 配置 ORT Session 参数
    session_options_.SetIntraOpNumThreads(4); // 开启4线程加速CPU推理
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 加载模型
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

    // 硬编码 YOLOv8 默认的输入输出节点名称（99%的YOLOv8模型都是这两个名字）
    input_node_names_ = {"images"};
    output_node_names_ = {"output0"};
}

cv::Mat YoloPose::letterbox(const cv::Mat& source, cv::Size target_size, float& ratio, int& pad_w, int& pad_h) {
    int src_w = source.cols;
    int src_h = source.rows;
    ratio = std::min((float)target_size.width / src_w, (float)target_size.height / src_h);
    int new_unpad_w = int(round(src_w * ratio));
    int new_unpad_h = int(round(src_h * ratio));
    pad_w = (target_size.width - new_unpad_w) / 2;
    pad_h = (target_size.height - new_unpad_h) / 2;

    cv::Mat unpad_img;
    cv::resize(source, unpad_img, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat padded_img;
    cv::copyMakeBorder(unpad_img, padded_img, pad_h, target_size.height - new_unpad_h - pad_h,
                       pad_w, target_size.width - new_unpad_w - pad_w, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return padded_img;
}

std::vector<ArmorObject> YoloPose::detect(cv::Mat& src) {
    std::vector<ArmorObject> results;
    if (src.empty()) return results;

    // 1. 预处理 (LetterBox 缩放并填充边界)
    float ratio;
    int pad_w, pad_h;
    cv::Mat input_img = letterbox(src, input_size_, ratio, pad_w, pad_h);

    // 2. 将图像转为张量格式 [1, 3, 640, 640]，利用 OpenCV 函数完成归一化和 RGB 转换
    cv::Mat blob;
    cv::dnn::blobFromImage(input_img, blob, 1.0 / 255.0, input_size_, cv::Scalar(), true, false);

    // 构建 ONNXRuntime 输入 Tensor
    std::vector<int64_t> input_shape = {1, 3, input_size_.height, input_size_.width};
    size_t input_tensor_size = 1 * 3 * input_size_.height * input_size_.width;
    
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, (float*)blob.data, input_tensor_size, input_shape.data(), input_shape.size()));

    // 3. 执行推理
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr}, input_node_names_.data(), input_tensors.data(), 1, output_node_names_.data(), 1);

    // 4. 获取输出数据
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    // YOLOv8 预测输出维度通常为 [1, dimensions, 8400]
    int dimensions = output_shape[1]; // 例如: 4(box) + classes + 4*3(kps)
    int rows = output_shape[2];       // 8400个预测框

    // 巧妙利用 OpenCV 矩阵转置：从 [dimensions, 8400] 转为 [8400, dimensions]
    cv::Mat pred_mat(dimensions, rows, CV_32F, output_data);
    cv::Mat pred = pred_mat.t(); 
    float* data = (float*)pred.data;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<std::vector<cv::Point2f>> all_keypoints;

    // 5. 遍历预测框，解析结果
    for (int i = 0; i < rows; ++i) {
        float* row_ptr = data + i * dimensions;

        // 查找最高置信度的类别
        float max_class_prob = 0.0f;
        int class_id = -1;
        for (int c = 0; c < num_classes_; ++c) {
            if (row_ptr[4 + c] > max_class_prob) {
                max_class_prob = row_ptr[4 + c];
                class_id = c;
            }
        }

        // 阈值过滤
        if (max_class_prob >= conf_thresh_) {
            float cx = row_ptr[0];
            float cy = row_ptr[1];
            float w = row_ptr[2];
            float h = row_ptr[3];
            
            int left = int((cx - 0.5 * w - pad_w) / ratio);
            int top = int((cy - 0.5 * h - pad_h) / ratio);
            int width = int(w / ratio);
            int height = int(h / ratio);
            
            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(max_class_prob);
            class_ids.push_back(class_id);

            // 解析4个角点
            std::vector<cv::Point2f> kps;
            int kp_index = 4 + num_classes_;
            for (int k = 0; k < 4; ++k) {
                float kx = (row_ptr[kp_index + k * 3] - pad_w) / ratio;
                float ky = (row_ptr[kp_index + k * 3 + 1] - pad_h) / ratio;
                kps.push_back(cv::Point2f(kx, ky));
            }
            all_keypoints.push_back(kps);
        }
    }

    // 6. NMS 非极大值抑制 (去除重复框)
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_thresh_, nms_thresh_, nms_indices);

    for (int idx : nms_indices) {
        ArmorObject armor;
        armor.box = boxes[idx];
        armor.confidence = confidences[idx];
        armor.class_id = class_ids[idx];
        armor.kps = all_keypoints[idx];
        results.push_back(armor);
    }

    return results;
}

void YoloPose::draw(cv::Mat& img, const std::vector<ArmorObject>& armors) {
    for (const auto& armor : armors) {
        cv::rectangle(img, armor.box, cv::Scalar(0, 255, 0), 2);
        std::string label = "Type: " + std::to_string(armor.class_id) + " (" + std::to_string(armor.confidence).substr(0,4) + ")";
        cv::putText(img, label, cv::Point(armor.box.x, armor.box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        if (armor.kps.size() == 4) {
            for (int i = 0; i < 4; ++i) {
                cv::circle(img, armor.kps[i], 4, cv::Scalar(0, 0, 255), -1);
                cv::line(img, armor.kps[i], armor.kps[(i + 1) % 4], cv::Scalar(255, 0, 0), 2);
            }
        }
    }
}