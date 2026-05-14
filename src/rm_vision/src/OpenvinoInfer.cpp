#include "OpenvinoInfer.h"
#include <algorithm>

OpenvinoArmorDetector::OpenvinoArmorDetector(const std::string& onnx_path,
                                             const std::string& device,
                                             float conf_thresh,
                                             float nms_thresh,
                                             int detect_color)
    : conf_threshold_(conf_thresh),
      nms_threshold_(nms_thresh),
      detect_color_(detect_color)
{
    // 加载 ONNX 模型
    model = core.read_model(onnx_path);

    // 设置预处理：输入为 640x640 BGR 图像
    ov::preprocess::PrePostProcessor ppp(model);
    // 告诉 OpenVINO 我们输入的 tensor 格式
    ppp.input().tensor()
        .set_element_type(ov::element::u8)
        .set_shape({1, INPUT_HEIGHT, INPUT_WIDTH, 3})   // NHWC
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);
    // 预处理步骤：缩放到模型需要的输入（这里假设模型输入就是 640x640，不需要 resize）
    // 如果你的模型允许动态尺寸，可以添加 .resize()
    ppp.input().preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .scale({255.0f, 255.0f, 255.0f});
    // 模型内部使用 NCHW
    ppp.input().model().set_layout("NCHW");

    // 输出保持 FP32
    ppp.output().tensor().set_element_type(ov::element::f32);

    // 应用预处理
    model = ppp.build();

    // 编译模型
    compiled_model = core.compile_model(model, device);
    // 创建推理请求（复用，不要每次重新创建）
    infer_request = compiled_model.create_infer_request();
}

cv::Mat OpenvinoArmorDetector::preprocess(const cv::Mat& img) {
    // 直接缩放到 640x640（和原来 blobFromImage 一样，会拉伸）
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
    return resized;
}

std::vector<DetectedArmor> OpenvinoArmorDetector::detect(const cv::Mat& img) {
    if (img.empty()) return {};

    // ---------- 预处理 ----------
    cv::Mat processed = preprocess(img);

    // ---------- 准备输入 tensor ----------
    // 注意：预处理在模型内部已完成 BGR->RGB 和除以 255，所以这里只需要提供原始 BGR 数据
    input_tensor = ov::Tensor(compiled_model.input().get_element_type(),
                              compiled_model.input().get_shape(),
                              processed.data);

    // ---------- 推理 ----------
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    // ---------- 获取输出 ----------
    const ov::Tensor& output_tensor = infer_request.get_output_tensor();
    const float* output_data = output_tensor.data<float>();
    ov::Shape output_shape = output_tensor.get_shape();

    // ---------- 后处理 ----------
    return postprocess(output_data, output_shape, img.cols, img.rows);
}

std::vector<DetectedArmor> OpenvinoArmorDetector::postprocess(
    const float* output_data,
    const ov::Shape& output_shape,
    int img_w, int img_h)
{
    // 输出形状：[1, num_anchors, 22]
    int num_anchors = output_shape[1];
    int num_features = output_shape[2];  // 22

    cv::Mat output_buffer(num_anchors, num_features, CV_32F, (void*)output_data);

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<DetectedArmor> temp_armors;

    float x_factor = img_w / (float)INPUT_WIDTH;
    float y_factor = img_h / (float)INPUT_HEIGHT;

    for (int i = 0; i < num_anchors; ++i) {
        // 置信度 + sigmoid
        float confidence = output_buffer.at<float>(i, 8);
        confidence = 1.0f / (1.0f + std::exp(-confidence));
        if (confidence < conf_threshold_) continue;

        // 颜色过滤（9~12）
        cv::Mat color_scores = output_buffer.row(i).colRange(9, 13);
        cv::Point color_id;
        double score_color;
        cv::minMaxLoc(color_scores, nullptr, &score_color, nullptr, &color_id);
        if (color_id.x == 2 || color_id.x == 3) continue;
        if (detect_color_ == 0 && color_id.x == 1) continue;
        if (detect_color_ == 1 && color_id.x == 0) continue;

        // 数字分类（13~21）
        cv::Mat class_scores = output_buffer.row(i).colRange(13, 22);
        cv::Point class_id;
        double score_num;
        cv::minMaxLoc(class_scores, nullptr, &score_num, nullptr, &class_id);

        // 还原四个角点坐标（0~7）
        cv::Point2f p1(output_buffer.at<float>(i, 0) * x_factor,
                       output_buffer.at<float>(i, 1) * y_factor);
        cv::Point2f p2(output_buffer.at<float>(i, 2) * x_factor,
                       output_buffer.at<float>(i, 3) * y_factor);
        cv::Point2f p3(output_buffer.at<float>(i, 4) * x_factor,
                       output_buffer.at<float>(i, 5) * y_factor);
        cv::Point2f p4(output_buffer.at<float>(i, 6) * x_factor,
                       output_buffer.at<float>(i, 7) * y_factor);

        // 计算水平外接矩形（用于 NMS）
        float x_min = std::min({p1.x, p2.x, p3.x, p4.x});
        float x_max = std::max({p1.x, p2.x, p3.x, p4.x});
        float y_min = std::min({p1.y, p2.y, p3.y, p4.y});
        float y_max = std::max({p1.y, p2.y, p3.y, p4.y});
        cv::Rect bounding_box(x_min, y_min, x_max - x_min, y_max - y_min);

        DetectedArmor armor;
        armor.pnp_corners = {p1, p2, p3, p4};
        armor.number = class_id.x;
        armor.class_confidence = static_cast<float>(score_num);
        armor.confidence = confidence;
        armor.rect = cv::minAreaRect(armor.pnp_corners);

        boxes.push_back(bounding_box);
        confidences.push_back(confidence);
        temp_armors.push_back(armor);
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);

    std::vector<DetectedArmor> final_armors;
    cv::Point2f img_center(img_w / 2.0f, img_h / 2.0f);
    for (int idx : indices) {
        DetectedArmor armor = temp_armors[idx];
        armor.dist_to_center = cv::norm(armor.rect.center - img_center);
        final_armors.push_back(armor);
    }
    return final_armors;
}