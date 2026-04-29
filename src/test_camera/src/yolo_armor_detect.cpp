#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>

// 包含自定义接口
#include "armor_interfaces/msg/armor.hpp"
#include "armor_interfaces/msg/armor_array.hpp"
#include "armor_interfaces/msg/serial.hpp"

// 包含功能组件
#include "Camera.hpp"
#include "YoloPose.hpp"
#include "Pnp.hpp"

struct GimbalState {
    rclcpp::Time stamp;
    double yaw;
    double pitch;
};

struct TrackedTarget {
    int id;
    double x, y, z;
    rclcpp::Time last_seen;
};

class YoloArmorDetectorNode : public rclcpp::Node
{
public:
    YoloArmorDetectorNode() : Node("yolo_armor_detector_node"), is_running_(true)
    {
        RCLCPP_INFO(this->get_logger(), "YOLO 端到端装甲板检测节点启动...");

        // 1. 初始化 ROS 参数
        this->declare_parameter<std::string>("model_path", "/home/aa/rm_ws/best.onnx");
        this->declare_parameter<int>("num_classes", 3);  // 根据你的数据集种类数量修改
        this->declare_parameter<float>("exposure", 3.0); // 曝光时间 ms

        std::string model_path = this->get_parameter("model_path").as_string();
        int num_classes = this->get_parameter("num_classes").as_int();
        float exposure = this->get_parameter("exposure").as_double();

        // 2. 初始化核心组件
        pnpsolver_ = std::make_unique<PnpSolver>();
        yolo_ = std::make_unique<YoloPose>(model_path, num_classes, 0.5, 0.4);
        camera_ = std::make_unique<CameraWrapper>();

        if (!camera_->init(0)) {
            RCLCPP_ERROR(this->get_logger(), "相机打开失败，节点退出");
            rclcpp::shutdown();
            return;
        }
        camera_->setExposure(exposure);

        // 3. 初始化 ROS 通信
        pub_ = this->create_publisher<armor_interfaces::msg::ArmorArray>("armor_msgs", 10);
        sub_ = this->create_subscription<armor_interfaces::msg::Serial>(
            "serial_data", 10, std::bind(&YoloArmorDetectorNode::gimbal_callback, this, std::placeholders::_1));

        // 4. 启动独立图像处理线程 (替代 timer_callback)
        capture_thread_ = std::thread(&YoloArmorDetectorNode::capture_loop, this);
    }

    ~YoloArmorDetectorNode()
    {
        is_running_ = false;
        if (capture_thread_.joinable()) {
            capture_thread_.join();
        }
    }

private:
    rclcpp::Publisher<armor_interfaces::msg::ArmorArray>::SharedPtr pub_;
    rclcpp::Subscription<armor_interfaces::msg::Serial>::SharedPtr sub_;

    std::unique_ptr<CameraWrapper> camera_;
    std::unique_ptr<YoloPose> yolo_;
    std::unique_ptr<PnpSolver> pnpsolver_;

    std::thread capture_thread_;
    std::atomic<bool> is_running_;

    // 时间同步队列与锁
    std::deque<GimbalState> gimbal_queue_;
    std::mutex queue_mutex_;

    // 追踪变量
    std::vector<TrackedTarget> prev_targets_;
    int next_id_ = 1;

    // 接收云台串口数据，塞入历史队列
    void gimbal_callback(const armor_interfaces::msg::Serial::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        GimbalState state;
        state.stamp = msg->header.stamp;
        state.yaw = msg->yaw;
        state.pitch = msg->pitch;
        
        gimbal_queue_.push_back(state);
        // 维持队列长度不超过 100 帧（防止内存泄露）
        if (gimbal_queue_.size() > 100) {
            gimbal_queue_.pop_front();
        }
    }

    // 根据图像时间戳，寻找最接近的云台绝对角度
    bool getClosestGimbalState(const rclcpp::Time& img_time, double& out_yaw, double& out_pitch)
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (gimbal_queue_.empty()) return false;

        double min_diff = 9999.0;
        GimbalState closest_state = gimbal_queue_.front();

        for (const auto& state : gimbal_queue_) {
            double diff = std::abs((state.stamp - img_time).seconds());
            if (diff < min_diff) {
                min_diff = diff;
                closest_state = state;
            }
        }

        // 容忍度：如果最近的串口数据也是 0.1 秒前的，说明通信断了，丢弃
        if (min_diff > 0.1) return false; 

        out_yaw = closest_state.yaw;
        out_pitch = closest_state.pitch;
        return true;
    }

    // 独立线程：相机出图就处理，毫无阻塞延迟
    void capture_loop()
    {
        cv::Mat img;
        while (is_running_ && rclcpp::ok()) 
        {
            // 1. 阻塞获取图像
            if (!camera_->getFrame(img, 1000)) {
                RCLCPP_WARN(this->get_logger(), "获取图像帧失败或超时");
                continue;
            }

            // 2. 图像获取成功瞬间，打上时间戳
            rclcpp::Time img_time = this->now();

            // 3. YOLO 模型推理
            std::vector<ArmorObject> detections = yolo_->detect(img);

            armor_interfaces::msg::ArmorArray armor_array_msg;
            armor_array_msg.header.stamp = img_time;
            armor_array_msg.header.frame_id = "camera";

            std::vector<TrackedTarget> curr_targets;

            // 4. 对获取到的时间戳寻找云台对应时刻的角度
            //double sync_gimbal_yaw = 0.0, sync_gimbal_pitch = 0.0;
            //bool is_synced = getClosestGimbalState(img_time, sync_gimbal_yaw, sync_gimbal_pitch);

            /*if (!is_synced) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "云台数据未对齐，本次解算可能不准！");
            }*/

            // 5. 遍历识别结果
            for (auto& det : detections) 
            {
                if (det.kps.size() != 4) continue;

                // --- 坐标对齐注意 ---
                // YoloPose 预测的4个角点顺序需要符合 PnP 期望的顺序 (一般是 左上->右上->右下->左下)
                // 如果在标注时就是按这个顺序标的，这里可以直接传
                cv::Mat rvec, tvec;
                double yaw_rel, pitch_rel, distance;
                
                if (pnpsolver_->solveWithPose(det.kps, rvec, tvec, yaw_rel, pitch_rel, distance)) 
                {
                    // 记录三维坐标（毫米单位用于欧氏距离追踪）
                    double target_x = tvec.at<double>(0) * 1000.0;
                    double target_y = tvec.at<double>(1) * 1000.0;
                    double target_z = tvec.at<double>(2) * 1000.0;

                    // --- 简单 ID 追踪逻辑 (防跳变) ---
                    int matched_id = -1;
                    double best_dist = 300.0;
                    for (const auto& prev_t : prev_targets_) {
                        double dx = target_x - prev_t.x;
                        double dy = target_y - prev_t.y;
                        double dist = std::sqrt(dx*dx + dy*dy);
                        if (dist < best_dist) {
                            best_dist = dist;
                            matched_id = prev_t.id;
                        }
                    }
                    if (matched_id == -1) matched_id = next_id_++;

                    // 保存当前追踪状态
                    TrackedTarget tt;
                    tt.id = matched_id;
                    tt.x = target_x; tt.y = target_y; tt.z = target_z;
                    tt.last_seen = img_time;
                    curr_targets.push_back(tt);

                    // --- 核心：计算绝对坐标 (视觉相对 + 云台历史绝对) ---
                    // 这里利用的是基于时间戳找出的云台角度，不会因为滞后而抽搐
                    double abs_yaw = yaw_rel;
                    double abs_pitch = pitch_rel;

                    // --- 封装消息 ---
                    armor_interfaces::msg::Armor armor_msg;
                    armor_msg.id = matched_id;          // 目标ID
                    //armor_msg.type = det.class_id;      // 装甲板种类 (需要修改 msg 添加 type 字段，或者你用别的方式传)
                    armor_msg.yaw = abs_yaw;            // 发送绝对水平角
                    armor_msg.pitch = abs_pitch;        // 发送绝对俯仰角
                    // 初始未滤波时，将其值设为一样
                    armor_msg.yaw_filtered = abs_yaw;
                    armor_msg.pitch_filtered = abs_pitch;
                    armor_msg.is_predict = false;

                    armor_array_msg.armors.push_back(armor_msg);

                    // --- 可视化打印 ---
                    cv::Point2f center = (det.kps[0] + det.kps[2]) / 2.0;
                    cv::putText(img, "ID: " + std::to_string(matched_id), center, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
                }
            }

            // 更新追踪列表 (清除太老的)
            prev_targets_.clear();
            for (const auto& tt : curr_targets) {
                if ((img_time - tt.last_seen).seconds() < 0.5) {
                    prev_targets_.push_back(tt);
                }
            }

            // 发布消息给卡尔曼节点
            if (!armor_array_msg.armors.empty()) {
                pub_->publish(armor_array_msg);
            }

            // 绘制画面 (YOLO画框与连线)
            yolo_->draw(img, detections);
            cv::imshow("YOLO Armor Detector", img);
            cv::waitKey(1); // 不要漏掉 waitKey，否则无法显示
        }
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<YoloArmorDetectorNode>());
    rclcpp::shutdown();
    return 0;
}