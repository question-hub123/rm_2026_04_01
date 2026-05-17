#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "armor_interfaces/msg/armor.hpp"
#include "armor_interfaces/msg/armor_array.hpp"
#include "armor_interfaces/msg/serial.hpp"

// 你的模块
#include "OpenvinoInfer.h"   // 需包含 DetectedArmor 等
#include "Pnp.hpp"
#include "EKF.hpp"
#include "Monitor.hpp"
#include "Tool.hpp"

using namespace std::chrono_literals;

class ArmorDetctor : public rclcpp::Node
{
public:
    ArmorDetctor() : Node("armor_detctor_node")
    {
        RCLCPP_INFO(this->get_logger(), "装甲板检测节点启动 (YOLO+EKF, 视频输入)");

        // ================= 参数声明 =================
        this->declare_parameter<std::string>("model_path", "/home/aa/rm_ws/0526.onnx");
        this->declare_parameter<std::string>("device", "CPU");
        this->declare_parameter<float>("conf_thresh", 0.65f);
        this->declare_parameter<float>("nms_thresh", 0.45f);
        this->declare_parameter<int>("detect_color", 0);
        this->declare_parameter<std::string>("video_path", "/home/aa/vision_source/test2.mp4");
        this->declare_parameter<bool>("show_window", true);

        // 获取参数
        std::string model_path = this->get_parameter("model_path").as_string();
        std::string device     = this->get_parameter("device").as_string();
        float conf_thresh      = this->get_parameter("conf_thresh").as_double();
        float nms_thresh       = this->get_parameter("nms_thresh").as_double();
        int detect_color       = this->get_parameter("detect_color").as_int();
        std::string video_path = this->get_parameter("video_path").as_string();
        show_window_           = this->get_parameter("show_window").as_bool();

        // ================= 模块初始化 =================
        // YOLO 检测器
        detector_   = std::make_unique<OpenvinoArmorDetector>(model_path, device, conf_thresh, nms_thresh, detect_color);
        pnp_solver_ = std::make_unique<PnpSolver>();

        // 视频读取
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

        // 定时器：按视频帧率驱动
        double fps = cap_.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 30.0;
        int period_ms = static_cast<int>(1000.0 / fps);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(period_ms),
            std::bind(&ArmorDetctor::timer_callback, this));

        // 时间戳初始化
        last_time_ = this->now();

        if (show_window_) {
            cv::namedWindow("Armor Detection", cv::WINDOW_AUTOSIZE);
        }
    }

private:
    // 通信
    rclcpp::Publisher<armor_interfaces::msg::ArmorArray>::SharedPtr pub_;
    rclcpp::Subscription<armor_interfaces::msg::Serial>::SharedPtr sub_;
    rclcpp::TimerBase::SharedPtr timer_;
    Tool tool_;

    // 视频
    cv::VideoCapture cap_;
    bool show_window_;

    // 模块
    std::unique_ptr<OpenvinoArmorDetector> detector_;
    std::unique_ptr<PnpSolver> pnp_solver_;

    // 云台角度（弧度）
    std::atomic<double> gimbal_yaw_{0.0};
    std::atomic<double> gimbal_pitch_{0.0};

    // 目标跟踪与 EKF
    struct TrackedTarget
    {
        int id;
        double x, y, z;          // 当前帧观测位置（相机系，米），用于匹配
        double yaw;              // 相对 yaw (rad)
        std::unique_ptr<EKF> ekf;
        bool ekf_initialized = false;
        rclcpp::Time last_seen;
    };
    std::vector<TrackedTarget> prev_targets_;
    int next_id_ = 1;
    rclcpp::Time last_time_;     // 上一帧时间戳（用于计算 dt）

    void sub_callback(const armor_interfaces::msg::Serial::SharedPtr msg)
    {
        gimbal_yaw_   = msg->yaw;    // 假设 msg 中为弧度
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

        // 时间步长
        rclcpp::Time now = this->now();
        double dt = (now - last_time_).seconds();
        last_time_ = now;

        // ===================== YOLO 检测 =====================
        auto armors = detector_->detect(frame);

        // 暂存本帧所有检测结果（相机系坐标，相对角度）
        struct Detection {
            cv::Point2f corners[4];   // 像素角点
            double x, y, z;           // 相机坐标系 (m)
            double yaw, pitch;        // 相对角度 (rad)，已取负适配云台方向
            int number;                 // 数字类别
        };
        std::vector<Detection> detections;

        // 遍历每个检测，做 PnP 解算
        for (const auto &armor : armors) {
            cv::Point2f vertices[4];
            armor.rect.points(vertices);

            // 角点重排以匹配 PnP 3D 点（左上、右上、右下、左下）
            std::vector<cv::Point2f> image_points = {
                vertices[0], vertices[3], vertices[2], vertices[1]
            };

            cv::Mat rvec, tvec;
            double yaw_rad, pitch_rad, distance;
            if (pnp_solver_->solveWithPose(image_points, rvec, tvec, yaw_rad, pitch_rad, distance)) {
                Detection det;

                Eigen::Vector3d pos_cam(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)); // 相机系位置
                Eigen::Vector3d pos_world = tool_.cameraToWorld(pos_cam, gimbal_yaw_.load(), gimbal_pitch_.load());

                for (int k = 0; k < 4; ++k) det.corners[k] = vertices[k];
                {
                    det.x     = pos_world.x();   // 米
                    det.y     = pos_world.y();
                    det.z     = pos_world.z();

                    double world_yaw, world_pitch;
                    tool_.cameraNormalToWorld(rvec, gimbal_yaw_.load(), gimbal_pitch_.load(), world_yaw, world_pitch);

                    det.yaw   = world_yaw;             // 注意符号，与原始模板保持一致
                    det.pitch = world_pitch;
                    det.number = armor.number;
                    detections.push_back(det);
                }
            }
        }

        // ===================== 目标关联（最近邻） =====================
        std::vector<int> assigned_ids(detections.size(), -1);
        std::vector<bool> used_prev(prev_targets_.size(), false);
        const double max_age = 0.5;  // 秒

        // 清除超时目标
        std::vector<TrackedTarget> alive;
        for (auto &t : prev_targets_) {
            if ((now - t.last_seen).seconds() < max_age) {
                alive.push_back(std::move(t));
            }
        }
        prev_targets_ = std::move(alive);

        // 匹配：基于 xy 平面距离
        for (size_t i = 0; i < detections.size(); ++i) {
            double best_dist = 0.3;   // 米
            int best_j = -1;
            for (size_t j = 0; j < prev_targets_.size(); ++j) {
                if (used_prev[j]) continue;
                double dx = detections[i].x - prev_targets_[j].x;
                double dy = detections[i].y - prev_targets_[j].y;
                double dist = std::sqrt(dx*dx + dy*dy);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_j = j;
                }
            }
            if (best_j >= 0) {
                assigned_ids[i] = prev_targets_[best_j].id;
                used_prev[best_j] = true;
            } else {
                assigned_ids[i] = next_id_++;
            }
        }

        // 更新本帧目标列表，并为每个目标运行 EKF
        std::vector<TrackedTarget> curr_targets;

        for (size_t i = 0; i < detections.size(); ++i) {
            // 查找或创建目标
            TrackedTarget tt;
            tt.id = assigned_ids[i];
            tt.x = detections[i].x;
            tt.y = detections[i].y;
            tt.z = detections[i].z;
            tt.yaw = detections[i].yaw;
            tt.last_seen = now;

            // 检查是否有旧的 EKF 实例
            auto old_it = std::find_if(prev_targets_.begin(), prev_targets_.end(),
                                       [&](const TrackedTarget &t) { return t.id == tt.id; });

            if (old_it != prev_targets_.end() && old_it->ekf) {
                // 继承 EKF 实例
                tt.ekf = std::move(old_it->ekf);
                tt.ekf_initialized = true;
            } else {
                // 创建新的 EKF，稍后初始化
                tt.ekf = std::make_unique<EKF>();
                tt.ekf_initialized = false;
            }

            // EKF 预测 + 更新
            if (tt.ekf_initialized) {
                tt.ekf->predict(dt);
            }
            // 观测值：相机系下的位置和 yaw
            Eigen::Vector3d pos_obs(tt.x, tt.y, tt.z);
            double yaw_obs = tt.yaw;
            if (!tt.ekf_initialized) {
                tt.ekf->init(pos_obs, yaw_obs);
                tt.ekf_initialized = true;
            } else {
                Eigen::Vector4d z;
                z << pos_obs.x(), pos_obs.y(), pos_obs.z(), yaw_obs;
                tt.ekf->update(z);
            }

            curr_targets.push_back(std::move(tt));
        }

        // 更新全局目标列表
        prev_targets_ = std::move(curr_targets);

        // ===================== 发布 ArmorArray =====================
        armor_interfaces::msg::ArmorArray armor_array_msg;
        armor_array_msg.header.stamp = now;
        armor_array_msg.header.frame_id = "camera";

        double cur_gimbal_yaw   = gimbal_yaw_.load();
        double cur_gimbal_pitch = gimbal_pitch_.load();


        for (size_t i = 0; i < detections.size(); ++i) {
            int id = assigned_ids[i];
            // 从 curr_targets 中获取 EKF 滤波后的状态（仅用于可能的滤波输出）
            auto it = std::find_if(prev_targets_.begin(), prev_targets_.end(),
                                   [&](const TrackedTarget &t) { return t.id == id; });

            double x_raw = detections[i].x;
            double y_raw = detections[i].y;
            double z_raw = detections[i].z;
            double yaw_rel = detections[i].yaw;
            double pitch_rel = detections[i].pitch;

            // 绝对角度
            double abs_yaw   = yaw_rel;
            double abs_pitch = pitch_rel;

            // 打包消息（单位：米，弧度）
            armor_interfaces::msg::Armor armor_msg;
            armor_msg.id              = detections[i].number; // 数字类别作为 ID，或使用 assigned_ids[i] 作为跟踪 ID
            armor_msg.x               = x_raw;         // 世界坐标下位置（米）
            armor_msg.y               = y_raw;
            armor_msg.z               = z_raw;
            
            armor_msg.yaw             = abs_yaw;        // 绝对角度（弧度）
            armor_msg.pitch           = abs_pitch;
            armor_msg.yaw_filtered    = abs_yaw;        // 未滤波，流程有点不完美，后续滤波值在EKF_node中完善
            armor_msg.pitch_filtered  = abs_pitch;
            armor_msg.is_predict      = false;

            // 可填充 EKF 给出的平滑值
            if (it != prev_targets_.end() && it->ekf_initialized) {
                Eigen::Vector3d pos_c;
                double yaw_est, v_yaw, r;
                pos_c = it->ekf->getVehiclePosition();   // Eigen::Vector3d
                yaw_est = it->ekf->getContinuousYaw();   // double
                v_yaw = it->ekf->x(7);                   // 偏航角速度
                r = it->ekf->getRadius();               // double
                // 这里 pos_c 是车体中心，不是装甲板位置，仅做演示，不做覆盖
            }

            armor_array_msg.armors.push_back(armor_msg);
            if(detections.empty()) { armor_array_msg.armors.clear(); }
        }

        pub_->publish(armor_array_msg);

        // ===================== 可视化 =====================
        if (show_window_) {
            for (size_t i = 0; i < detections.size(); ++i) {
                const Detection &det = detections[i];
                int id = assigned_ids[i];

                // 画旋转矩形（四点连线）
                for (int k = 0; k < 4; ++k) {
                    cv::line(frame, det.corners[k], det.corners[(k+1)%4], cv::Scalar(0,255,0), 2);
                }

                // 中心点
                cv::Point2f center = (det.corners[0] + det.corners[2]) / 2.0f;
                
                cv::putText(frame, "Num:" + std::to_string(det.number), center,
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0,255,0), 3);

                // 距离与角度
                double dist_m = std::sqrt(det.x*det.x + det.y*det.y + det.z*det.z);
                cv::putText(frame, cv::format("Dist:%.2fm", dist_m),
                            center + cv::Point2f(-80, 20),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255), 2);
                cv::putText(frame, cv::format("Y:%.1f P:%.1f",
                                              det.yaw*180/M_PI, det.pitch*180/M_PI),
                            center + cv::Point2f(-80, 45),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,0), 2);

                // 若 EKF 已初始化，显示其车体估计（仅打印）
                auto it = std::find_if(prev_targets_.begin(), prev_targets_.end(),
                                       [&](const TrackedTarget &t) { return t.id == id; });
                if (it != prev_targets_.end() && it->ekf_initialized) {
                    Eigen::Vector3d pos_c;
                    double yaw_est, v_yaw, r;
                    pos_c = it->ekf->getVehiclePosition();   // Eigen::Vector3d
                    yaw_est = it->ekf->getContinuousYaw();   // double
                    v_yaw = it->ekf->x(7);                   // 偏航角速度
                    r = it->ekf->getRadius();               // double

                    /*tool_.drawAllArmors(frame, pos_c, yaw_est, r, det.number);
                    cv::putText(frame, cv::format("EKFX:%.2f Y:%.2f Yaw:%.1f", pos_c.x(), pos_c.y(), yaw_est*180/M_PI),
                                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                                0.5, cv::Scalar(0,255,255), 1);*/
                }
            }

            // 显示推理耗时（大致 fps）
            cv::putText(frame, "YOLO+EKF", cv::Point(10, frame.rows - 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 1);
            cv::imshow("Armor Detection", frame);
            if (cv::waitKey(1) == 27) 
            {
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