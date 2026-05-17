#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "armor_interfaces/msg/armor.hpp"
#include "armor_interfaces/msg/armor_array.hpp"
#include "armor_interfaces/msg/serial.hpp"

#include "OpenvinoInfer.h"
#include "Pnp.hpp"
#include "Tool.hpp"

using namespace std::chrono_literals;

class ArmorDetctor : public rclcpp::Node
{
public:
    ArmorDetctor() : Node("armor_detctor_node")
    {
        RCLCPP_INFO(this->get_logger(), "装甲板检测节点启动 (仅检测+观测发布)");

        // ================= 参数声明 =================
        this->declare_parameter<std::string>("model_path", "/home/aa/rm_ws/0526.onnx");
        this->declare_parameter<std::string>("device", "CPU");
        this->declare_parameter<float>("conf_thresh", 0.65f);
        this->declare_parameter<float>("nms_thresh", 0.45f);
        this->declare_parameter<int>("detect_color", 0);
        this->declare_parameter<std::string>("video_path", "/home/aa/vision_source/test2.mp4");
        this->declare_parameter<bool>("show_window", true);

        std::string model_path = this->get_parameter("model_path").as_string();
        std::string device     = this->get_parameter("device").as_string();
        float conf_thresh      = this->get_parameter("conf_thresh").as_double();
        float nms_thresh       = this->get_parameter("nms_thresh").as_double();
        int detect_color       = this->get_parameter("detect_color").as_int();
        std::string video_path = this->get_parameter("video_path").as_string();
        show_window_           = this->get_parameter("show_window").as_bool();

        // ================= 模块初始化 =================
        detector_   = std::make_unique<OpenvinoArmorDetector>(model_path, device, conf_thresh, nms_thresh, detect_color);
        pnp_solver_ = std::make_unique<PnpSolver>();

        cap_.open(video_path);
        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "视频打开失败: %s", video_path.c_str());
            rclcpp::shutdown();
            return;
        }

        // ================= ROS 通信 =================
        pub_ = this->create_publisher<armor_interfaces::msg::ArmorArray>("armor_msgs", 10);
        sub_ = this->create_subscription<armor_interfaces::msg::Serial>(
            "serial_data", 10,
            std::bind(&ArmorDetctor::sub_callback, this, std::placeholders::_1));

        // 订阅滤波后的车体中心 (来自 EKF_node)
        sub_filtered_ = this->create_subscription<armor_interfaces::msg::ArmorArray>(
            "armor_msgs_filtered", 10,
            [this](const armor_interfaces::msg::ArmorArray::SharedPtr msg) {
                filtered_vehicle_centers_.clear();
                for (const auto& arm : msg->armors) {
                    // arm.x, arm.y, arm.z 已经是车体中心 (世界系)
                    filtered_vehicle_centers_.push_back(
                        Eigen::Vector3d(arm.x, arm.y, arm.z));
                }
            });

        // 定时器：按视频帧率驱动
        double fps = cap_.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 30.0;
        int period_ms = static_cast<int>(1000.0 / fps);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(period_ms),
            std::bind(&ArmorDetctor::timer_callback, this));

        last_time_ = this->now();

        if (show_window_) {
            cv::namedWindow("Armor Detection", cv::WINDOW_AUTOSIZE);
        }
    }

private:
    rclcpp::Publisher<armor_interfaces::msg::ArmorArray>::SharedPtr pub_;
    rclcpp::Subscription<armor_interfaces::msg::Serial>::SharedPtr sub_;
    rclcpp::Subscription<armor_interfaces::msg::ArmorArray>::SharedPtr sub_filtered_;
    rclcpp::TimerBase::SharedPtr timer_;
    Tool tool_;

    cv::VideoCapture cap_;
    bool show_window_;

    std::unique_ptr<OpenvinoArmorDetector> detector_;
    std::unique_ptr<PnpSolver> pnp_solver_;

    std::atomic<double> gimbal_yaw_{0.0};
    std::atomic<double> gimbal_pitch_{0.0};

    std::vector<Eigen::Vector3d> filtered_vehicle_centers_;  // 世界系

    rclcpp::Time last_time_;

    void sub_callback(const armor_interfaces::msg::Serial::SharedPtr msg)
    {
        gimbal_yaw_   = msg->yaw;
        gimbal_pitch_ = msg->pitch;
    }

    void timer_callback()
    {
        cv::Mat frame;
        cap_ >> frame;
        if (frame.empty()) {
            RCLCPP_INFO(this->get_logger(), "视频结束");
            rclcpp::shutdown();
            return;
        }

        rclcpp::Time now = this->now();
        double dt = (now - last_time_).seconds();
        last_time_ = now;

        // ===================== YOLO 检测 + PnP 解算 =====================
        auto armors = detector_->detect(frame);

        struct Detection {
            cv::Point2f corners[4];
            double x, y, z;       // 世界系坐标 (m)
            double yaw, pitch;    // 世界系姿态角 (rad)
            int number;
        };
        std::vector<Detection> detections;

        for (const auto &armor : armors) {
            cv::Point2f vertices[4];
            armor.rect.points(vertices);

            std::vector<cv::Point2f> image_points = {
                vertices[0], vertices[3], vertices[2], vertices[1]
            };

            cv::Mat rvec, tvec;
            double yaw_rad, pitch_rad, distance;
            if (pnp_solver_->solveWithPose(image_points, rvec, tvec, yaw_rad, pitch_rad, distance)) {
                Detection det;
                for (int k = 0; k < 4; ++k) det.corners[k] = vertices[k];

                // 世界系坐标
                Eigen::Vector3d pos_cam(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
                Eigen::Vector3d pos_world = tool_.cameraToWorld(pos_cam, gimbal_yaw_.load(), gimbal_pitch_.load());
                det.x = pos_world.x();
                det.y = pos_world.y();
                det.z = pos_world.z();

                // 世界系姿态
                double world_yaw, world_pitch;
                tool_.cameraNormalToWorld(rvec, gimbal_yaw_.load(), gimbal_pitch_.load(), world_yaw, world_pitch);
                det.yaw   = world_yaw;
                det.pitch = world_pitch;
                det.number = armor.number;
                detections.push_back(det);
            }
        }

        // ===================== 发布原始观测 =====================
        armor_interfaces::msg::ArmorArray armor_array_msg;
        armor_array_msg.header.stamp = now;
        armor_array_msg.header.frame_id = "camera";

        for (const auto& det : detections) {
            armor_interfaces::msg::Armor armor_msg;
            armor_msg.id              = det.number;
            armor_msg.x               = det.x;
            armor_msg.y               = det.y;
            armor_msg.z               = det.z;
            armor_msg.yaw             = det.yaw;
            armor_msg.pitch           = det.pitch;
            // 滤波字段留空或设为原始值（后续由 EKF_node 填充）
            armor_msg.yaw_filtered    = det.yaw;
            armor_msg.pitch_filtered  = det.pitch;
            armor_msg.is_predict      = false;
            armor_array_msg.armors.push_back(armor_msg);
        }
        pub_->publish(armor_array_msg);

        // ===================== 可视化 =====================
        if (show_window_) {
            // 1. 绘制检测结果
            for (const auto& det : detections) {
                // 画四边形
                for (int k = 0; k < 4; ++k)
                    cv::line(frame, det.corners[k], det.corners[(k+1)%4], cv::Scalar(0,255,0), 2);
                cv::Point2f center = (det.corners[0] + det.corners[2]) / 2.0f;
                cv::putText(frame, "Num:" + std::to_string(det.number), center,
                            cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0,255,0), 3);

                double dist_m = std::sqrt(det.x*det.x + det.y*det.y + det.z*det.z);
                cv::putText(frame, cv::format("Dist:%.2fm", dist_m),
                            center + cv::Point2f(-80, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255), 2);
                cv::putText(frame, cv::format("Y:%.1f P:%.1f", det.yaw*180/M_PI, det.pitch*180/M_PI),
                            center + cv::Point2f(-80, 45), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,0), 2);
            }

            // 2. 绘制滤波后的车体中心 (来自 EKF_node)
            for (const auto& center_world : filtered_vehicle_centers_) {
                tool_.drawVehicleCenter(frame, center_world,0,0);
            }

            // 3. 帧信息
            cv::putText(frame, "YOLO + EKF (remote)", cv::Point(10, frame.rows - 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 1);
            cv::imshow("Armor Detection", frame);
            if (cv::waitKey(1) == 27) {
                rclcpp::shutdown();
            }
        }
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArmorDetctor>());
    rclcpp::shutdown();
    return 0;
}