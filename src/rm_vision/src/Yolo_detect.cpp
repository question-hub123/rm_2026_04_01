#include <cstddef>
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
#include "yolo_tool.hpp"

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
        this->declare_parameter<float>("conf_thresh", 0.80f);
        this->declare_parameter<float>("nms_thresh", 0.45f);
        this->declare_parameter<int>("detect_color", 0);
        this->declare_parameter<std::string>("video_path", "/home/aa/vision_source/move.mp4");
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
                orientation_yaw.clear();
                for (const auto& arm : msg->armors) 
                {
                    // arm.x, arm.y, arm.z 已经是车体中心 (世界系)
                    filtered_vehicle_centers_.push_back(Eigen::Vector3d(arm.x, arm.y, arm.z));
                    orientation_yaw.push_back(arm.yaw);
                    R.push_back(arm.yaw_filtered); // 这里我们把 EKF 输出的角速度 R 存在了 yaw_filtered 字段里

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
    Yolo_Tool tool_;

    cv::VideoCapture cap_;
    bool show_window_;

    std::unique_ptr<OpenvinoArmorDetector> detector_;
    std::unique_ptr<PnpSolver> pnp_solver_;

    std::atomic<double> gimbal_yaw_{0.0};
    std::atomic<double> gimbal_pitch_{0.0};

    std::vector<Eigen::Vector3d> filtered_vehicle_centers_;  // 世界系
    std::vector<double> orientation_yaw;//装甲板朝向角
    std::vector<double> R;//装甲板半径 r（用 EKF 输出的 yaw_filtered 字段传递）

    bool is_paused_ = false;
    cv::Mat last_drawn_frame_;   // 存放最后一帧的绘制结果

    rclcpp::Time last_time_;

    void sub_callback(const armor_interfaces::msg::Serial::SharedPtr msg)
    {
        gimbal_yaw_   = msg->yaw;
        gimbal_pitch_ = msg->pitch;
    }

    void timer_callback()
    {
        if (is_paused_) 
        {
            if (!last_drawn_frame_.empty()) 
            {
                cv::Mat paused_display = last_drawn_frame_.clone();
                cv::putText(paused_display, "PAUSED", cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
                cv::imshow("Armor Detection", paused_display);
            }
            int key = cv::waitKey(1);
            if (key == 32) 
            {          // 空格键 → 恢复播放
                is_paused_ = false;
            } 
            else if (key == 27) 
            {   // ESC → 退出
                rclcpp::shutdown();
            }
            return;
        }


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
            double yaw, pitch, roll;    // 世界系姿态角 (rad)
            int number;
        };

        std::vector<Detection> detections;
        /*double pitch_est = 4.0 * M_PI / 180.0; // 约15度俯视
        Eigen::Quaterniond q_imu = Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ())
                         * Eigen::AngleAxisd(pitch_est, Eigen::Vector3d::UnitY());*/

        Eigen::Quaterniond q_imu = Eigen::Quaterniond::Identity();

        for (const auto &armor : armors) {
            cv::Point2f vertices[4];
            armor.rect.points(vertices);

            std::vector<cv::Point2f> raw_points(vertices, vertices + 4);
            std::vector<cv::Point2f> image_points = tool_.orderPoints(raw_points);

            cv::Mat rvec, tvec;
            double yaw_rad, pitch_rad, distance;
            if (pnp_solver_->solveWithPose(image_points, rvec, tvec, yaw_rad, pitch_rad, distance)) {
                Detection det;
                for (int k = 0; k < 4; ++k) det.corners[k] = image_points[k];

                Eigen::Vector3d pos_cam(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
                Eigen::Vector3d pos_world = tool_.cameraToWorld(pos_cam, q_imu);

                det.x = pos_world.x();
                det.y = pos_world.y();
                det.z = pos_world.z();

                double world_yaw, world_pitch, world_roll;
                tool_.cameraNormalToWorld(rvec, q_imu, world_yaw, world_pitch, world_roll);
                det.yaw   = world_yaw;
                det.pitch = world_pitch;
                det.roll  = world_roll;
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

            armor_msg.yaw_filtered    = det.roll;//懒得改信息包了，先对付一下
            armor_msg.pitch_filtered  = det.pitch;
            armor_msg.is_predict      = false;
            armor_array_msg.armors.push_back(armor_msg);

            double cx = det.x + 0.37 * sin(det.yaw);
            double cy = det.y + 0.37 * cos(det.yaw); 
            double cz = det.z;
            //tool_.drawVehicleCenter(frame, Eigen::Vector3d(cx, cy, cz), q_imu);
        }
        pub_->publish(armor_array_msg);

        // ===================== 可视化 =====================
        if (show_window_) {
            // 1. 绘制检测结果
            for (const auto& det : detections) {
                // 画四边形
                int num = 0;
                for (int k = 0; k < 4; ++k)
                {
                    cv::line(frame, det.corners[k], det.corners[(k+1)%4], cv::Scalar(0,255,0), 2);
                    cv::putText(frame, std::to_string(num++), det.corners[k], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 2);
                }
                cv::Point2f center = (det.corners[0] + det.corners[2]) / 2.0f;
                cv::putText(frame, "Num:" + std::to_string(det.number), center,
                            cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0,255,0), 3);

                double dist_m = std::sqrt(det.x*det.x + det.y*det.y + det.z*det.z);
                //cv::putText(frame, cv::format("Dist:%.2fm", dist_m),
                            //center + cv::Point2f(-80, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255), 2);
                //cv::putText(frame, cv::format("Y:%.1f P:%.1f R:%.1f", det.yaw*180/M_PI, det.pitch*180/M_PI, det.roll*180/M_PI),
                            //center + cv::Point2f(-80, 45), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,0), 2);
            }

            // 2. 绘制滤波后的车体中心 (来自 EKF_node）
            Eigen::Vector3d test = Eigen::Vector3d(2.16, -0.07, -0.11);
            for(size_t i = 0; i < filtered_vehicle_centers_.size(); ++i)
            {
                Eigen::Vector3d center_world = filtered_vehicle_centers_[i];
                double yaw = orientation_yaw[i];
                double r = R[i];
                double delay_s = 0.04;

                //yaw += Vyaw * delay_s; // 预测短时间后的朝向，补偿系统延迟
                
                tool_.drawVehicleCenter(frame, center_world, q_imu);
                tool_.drawAllArmors(frame, center_world, yaw, q_imu, r);
            }
            // 3. 帧信息
            cv::putText(frame, "YOLO + EKF (remote)", cv::Point(10, frame.rows - 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 1);
            cv::imshow("Armor Detection", frame);
            int key = cv::waitKey(10);
            if (key == 27) {         // ESC → 退出
                rclcpp::shutdown();
            } else if (key == 32) {  // 空格 → 暂停
                is_paused_ = true;
                last_drawn_frame_ = frame.clone();   // 保存绘制好的帧
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