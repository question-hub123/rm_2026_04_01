#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <vector>
#include <algorithm>

// ==================== 可调参数 ====================
const std::string MODEL_PATH = "best.onnx";   // ONNX 模型路径
const int INPUT_W  = 640;
const int INPUT_H  = 640;
const int NUM_CLASSES = 2;                    // ★ 按你的实际类别数修改
const float CONF_THRESH = 0.5f;
const float IOU_THRESH  = 0.45f;

// ★ 类别名（顺序严格与训练时的 names 一致）
const std::vector<std::string> CLASS_NAMES = { "red_armor", "blue_armor" };

// ==================== 检测结果 ====================
struct Detection {
    cv::Rect2f box;
    float conf;
    int class_id;
};

// ==================== YOLOv5 解码（已解码输出）====================
// 假设模型输出 shape 为 [1, 25200, 5+NUM_CLASSES]，坐标已是绝对像素值
void decodeYOLOv5(const cv::Mat& output, float scale_x, float scale_y,
                  std::vector<Detection>& dets)
{
    dets.clear();
    const float* data = reinterpret_cast<float*>(output.data);
    int rows = output.size[1];      // 25200
    int cols = output.size[2];      // 5 + NUM_CLASSES

    for (int i = 0; i < rows; ++i) {
        const float* row = data + i * cols;
        float obj_conf = row[4];
        if (obj_conf < CONF_THRESH) continue;

        // 找最大类别得分
        int best_cls = 0;
        float max_cls_score = 0.0f;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float s = row[5 + c];
            if (s > max_cls_score) {
                max_cls_score = s;
                best_cls = c;
            }
        }
        float conf = obj_conf * max_cls_score;
        if (conf < CONF_THRESH) continue;

        // 边界框 (cx,cy,w,h) -> (x1,y1,x2,y2)
        float cx = row[0], cy = row[1], w = row[2], h = row[3];
        float x1 = (cx - w / 2) * scale_x;
        float y1 = (cy - h / 2) * scale_y;
        float x2 = (cx + w / 2) * scale_x;
        float y2 = (cy + h / 2) * scale_y;

        // 裁剪到画面内
        x1 = std::max(0.0f, x1); y1 = std::max(0.0f, y1);
        x2 = std::min(x2, (float)(INPUT_W * scale_x));
        y2 = std::min(y2, (float)(INPUT_H * scale_y));

        if (x2 - x1 < 2 || y2 - y1 < 2) continue;

        dets.push_back({ cv::Rect2f(x1, y1, x2 - x1, y2 - y1), conf, best_cls });
    }
}

// ==================== NMS ====================
void applyNMS(std::vector<Detection>& dets, float iou_thres) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.conf > b.conf; });

    std::vector<bool> keep(dets.size(), true);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (!keep[i]) continue;
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!keep[j]) continue;
            float inter = (dets[i].box & dets[j].box).area();
            float iou = inter / (dets[i].box.area() + dets[j].box.area() - inter);
            if (iou > iou_thres) keep[j] = false;
        }
    }

    std::vector<Detection> nms_dets;
    for (size_t i = 0; i < dets.size(); ++i)
        if (keep[i]) nms_dets.push_back(dets[i]);
    dets.swap(nms_dets);
}

// ==================== 绘制 ====================
void drawResults(cv::Mat& frame, const std::vector<Detection>& dets) {
    for (const auto& d : dets) {
        cv::rectangle(frame, d.box, cv::Scalar(0, 255, 0), 2);
        std::string label = CLASS_NAMES[d.class_id] + " " + cv::format("%.2f", d.conf);
        int baseline;
        cv::Size txt = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(frame,
                      cv::Point(d.box.x, d.box.y - txt.height - 4),
                      cv::Point(d.box.x + txt.width, d.box.y),
                      cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(frame, label,
                    cv::Point(d.box.x, d.box.y - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

// ==================== 主函数 ====================
int main() {
    // 1. 加载 ONNX 模型
    cv::dnn::Net net = cv::dnn::readNetFromONNX(MODEL_PATH);
    // 使用 CPU 后端（OpenCV 4.5 默认）
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // 2. 打开摄像头
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera!" << std::endl;
        return -1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    cv::Mat frame, blob;
    cv::TickMeter timer;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        cv::flip(frame, frame, 1);   // 水平镜像（更自然）

        timer.reset();
        timer.start();

        // 3. 预处理：创建 blob (缩放、归一化、BGR->RGB)
        blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(INPUT_W, INPUT_H),
                                      cv::Scalar(0, 0, 0), true, false);

        // 4. 推理
        net.setInput(blob);
        cv::Mat output = net.forward();   // shape: [1, 25200, 5+NUM_CLASSES]

        // 5. 解码
        std::vector<Detection> detections;
        float scale_x = (float)frame.cols / INPUT_W;
        float scale_y = (float)frame.rows / INPUT_H;
        decodeYOLOv5(output, scale_x, scale_y, detections);

        // 6. NMS
        applyNMS(detections, IOU_THRESH);

        timer.stop();

        // 7. 绘制
        drawResults(frame, detections);
        std::string fps = cv::format("FPS: %.1f", timer.getFPS());
        cv::putText(frame, fps, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        cv::imshow("YOLO Armor Detection", frame);
        if (cv::waitKey(1) == 27) break;   // ESC 退出
    }

    return 0;
}