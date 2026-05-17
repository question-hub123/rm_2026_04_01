#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include "Monitor.hpp"
#include "armor_interfaces/msg/armor_array.hpp"
#include "armor_interfaces/msg/serial.hpp"
#include "armor_interfaces/msg/armor.hpp"
#include "EKF.hpp"
#include <deque>
#include <float.h>
#include <cmath>

using namespace std::chrono_literals;
using std::placeholders::_1;

class EKFNode : public rclcpp::Node {
public:
    EKFNode() : Node("ekf_node"), tracker_state_(LOST), tracked_id_(-1),
                detect_count_(0), lost_count_(0), last_yaw_(0.0), another_r_(0.26), dz_(0.0) {
        // 参数
        this->declare_parameter<int>("tracking_thres", 5);
        this->declare_parameter<int>("lost_thres", 15);
        this->declare_parameter<double>("max_match_distance", 1.5);
        this->declare_parameter<double>("max_match_yaw_diff", 0.8); // rad (~45°)
        tracking_thres_ = this->get_parameter("tracking_thres").as_int();
        lost_thres_ = this->get_parameter("lost_thres").as_int();
        max_match_distance_ = this->get_parameter("max_match_distance").as_double();
        max_match_yaw_diff_ = this->get_parameter("max_match_yaw_diff").as_double();

        // 订阅
        sub_armor_ = this->create_subscription<armor_interfaces::msg::ArmorArray>(
            "armor_msgs", 10, std::bind(&EKFNode::armorCallback, this, _1));
        sub_serial_ = this->create_subscription<armor_interfaces::msg::Serial>(
            "serial_data", 10, [this](const armor_interfaces::msg::Serial::SharedPtr msg) {
                last_gimbal_yaw_ = msg->yaw;
                last_gimbal_pitch_ = msg->pitch;
            });
        pub_filtered_ = this->create_publisher<armor_interfaces::msg::ArmorArray>(
            "armor_msgs_filtered", 10);

        // 可视化定时器
        plot_monitor_.initAnglePlot();
        gui_timer_ = this->create_wall_timer(10ms, [this]() { cv::waitKey(1); });

        RCLCPP_INFO(this->get_logger(), "EKF Node (9D) started.");
    }

private:
    enum State { LOST, DETECTING, TRACKING, TEMP_LOST };
    State tracker_state_;
    int tracked_id_;
    int detect_count_;
    int lost_count_;
    int tracking_thres_;
    int lost_thres_;
    double max_match_distance_;
    double max_match_yaw_diff_;
    double last_yaw_; // 用于连续化观测yaw
    double another_r_; // 用于NORMAL_4跳变时交换半径
    double dz_;        // 高度差 (用于NORMAL_4跳变)
    EKF ekf_;
    Monitor plot_monitor_;

    rclcpp::Subscription<armor_interfaces::msg::ArmorArray>::SharedPtr sub_armor_;
    rclcpp::Subscription<armor_interfaces::msg::Serial>::SharedPtr sub_serial_;
    rclcpp::Publisher<armor_interfaces::msg::ArmorArray>::SharedPtr pub_filtered_;
    rclcpp::TimerBase::SharedPtr gui_timer_;
    double last_gimbal_yaw_ = 0.0;
    double last_gimbal_pitch_ = 0.0;
    rclcpp::Time last_time_;

    // 工具函数：获取连续化的装甲板yaw (基于世界系姿态角)
    double getContinuousArmorYaw(const armor_interfaces::msg::Armor& armor) {
        // armor.yaw 是 PnP 解算出的装甲板自身姿态角 (世界系, -pi~pi)
        double yaw = armor.yaw;
        yaw = last_yaw_ + EKF::shortestAngularDistance(last_yaw_, yaw);
        last_yaw_ = yaw;
        return yaw;
    }

    // 从状态反推当前追踪的装甲板位置 (世界系)
    Eigen::Vector3d getPredictedArmorPosition(const EKF& ekf) {
        return ekf.getArmorPosition();
    }

    // 跳变处理 (小陀螺装甲板切换)
    void handleArmorJump(const armor_interfaces::msg::Armor& current_armor) {
        double yaw_continuous = getContinuousArmorYaw(current_armor);
        // 更新车体朝向
        ekf_.x(6) = yaw_continuous;
        // 对于普通4板布局，可能需要交换半径 (因为前后板的旋转半径可能不同)
        // 这里我们默认所有目标都是普通4板，或者通过判断 armor.type 来决定
        // 简单起见，我们假设总是 NORMAL_4，除非你能获取装甲板类型
        bool is_normal_4 = true; // 实际应根据检测到的车辆类型判断
        if (is_normal_4) {
            dz_ = ekf_.x(4) - current_armor.z; // 当前高度 - 之前高度
            ekf_.x(4) = current_armor.z;       // 更新高度
            std::swap(ekf_.x(8), another_r_);   // 交换半径
        }
        // 检查是否发散：若预测位置与当前观测差距过大，重置状态
        Eigen::Vector3d cur_pos(current_armor.x, current_armor.y, current_armor.z);
        Eigen::Vector3d pred_pos = getPredictedArmorPosition(ekf_);
        if ((cur_pos - pred_pos).norm() > max_match_distance_) {
            double r = ekf_.x(8);
            ekf_.x(0) = cur_pos.x() - r * cos(yaw_continuous);
            ekf_.x(2) = cur_pos.y() - r * sin(yaw_continuous);
            ekf_.x(4) = cur_pos.z();
            ekf_.x(6) = yaw_continuous;
            RCLCPP_WARN(this->get_logger(), "EKF diverged, reset state.");
        }
    }

    void armorCallback(const armor_interfaces::msg::ArmorArray::SharedPtr msg) {
        if (msg->armors.empty()) return;

        rclcpp::Time now = msg->header.stamp;
        if (last_time_.nanoseconds() == 0) {
            last_time_ = now;
            return;
        }
        double dt = (now - last_time_).seconds();
        last_time_ = now;
        if (dt <= 0.0 || dt > 0.5) dt = 0.01; // 保护

        // 状态机: 如果丢失，寻找新目标
        if (tracker_state_ == LOST) {
            // 选择距离图像中心最近的目标
            double min_dist = DBL_MAX;
            int best_idx = -1;
            for (size_t i = 0; i < msg->armors.size(); ++i) {
                // 假设 Armor 消息中有 distance_to_image_center 字段，如果没有，可以自己计算
                double d = std::sqrt(msg->armors[i].x * msg->armors[i].x + msg->armors[i].y * msg->armors[i].y);
                if (d < min_dist) {
                    min_dist = d;
                    best_idx = i;
                }
            }
            if (best_idx >= 0) {
                const auto& armor = msg->armors[best_idx];
                tracked_id_ = armor.id; // 数字ID
                double yaw_cont = getContinuousArmorYaw(armor);
                Eigen::Vector3d p_armor(armor.x, armor.y, armor.z);
                ekf_.init(p_armor, yaw_cont);
                tracker_state_ = DETECTING;
                detect_count_ = 0;
                RCLCPP_INFO(this->get_logger(), "Init tracking ID: %d", tracked_id_);
            }
            return;
        }

        // 预测
        ekf_.predict(dt);
        Eigen::Vector3d pred_armor_pos = getPredictedArmorPosition(ekf_);

        // 筛选同 ID 的装甲板
        std::vector<armor_interfaces::msg::Armor> candidates;
        for (const auto& arm : msg->armors) {
            if (arm.id == tracked_id_) candidates.push_back(arm);
        }

        bool matched = false;
        armor_interfaces::msg::Armor matched_armor;
        if (!candidates.empty()) {
            double min_pos_diff = DBL_MAX;
            double best_yaw_diff = 0.0;
            int best_cand = -1;
            for (size_t i = 0; i < candidates.size(); ++i) {
                const auto& arm = candidates[i];
                Eigen::Vector3d pos(arm.x, arm.y, arm.z);
                double pos_diff = (pos - pred_armor_pos).norm();
                double yaw_cont = getContinuousArmorYaw(arm);
                double yaw_diff = std::abs(EKF::shortestAngularDistance(yaw_cont, ekf_.getContinuousYaw()));
                if (pos_diff < min_pos_diff) {
                    min_pos_diff = pos_diff;
                    best_yaw_diff = yaw_diff;
                    best_cand = i;
                }
            }
            // 检查匹配条件
            if (min_pos_diff < max_match_distance_ && best_yaw_diff < max_match_yaw_diff_) {
                matched = true;
                matched_armor = candidates[best_cand];
                // 更新 EKF
                double obs_yaw = std::atan2(matched_armor.y, matched_armor.x);
                double obs_pitch = std::atan2(matched_armor.z, std::sqrt(matched_armor.x*matched_armor.x + matched_armor.y*matched_armor.y));
                double obs_dist = std::sqrt(matched_armor.x*matched_armor.x + matched_armor.y*matched_armor.y + matched_armor.z*matched_armor.z);
                double obs_armor_yaw = getContinuousArmorYaw(matched_armor);
                Eigen::Vector4d z(obs_yaw, obs_pitch, obs_dist, obs_armor_yaw);
                ekf_.update(z);

            } 
            else if (candidates.size() == 1 && best_yaw_diff > max_match_yaw_diff_) {
                // 疑似跳变: 只有一个同ID目标, 位置差小但yaw差大
                handleArmorJump(candidates[0]);
            }
        }

        // 半径钳制 (防止估计超出物理可能范围)
        if (ekf_.x(8) < 0.12) ekf_.x(8) = 0.12;
        if (ekf_.x(8) > 0.4)  ekf_.x(8) = 0.4;

        // 状态机更新
        if (tracker_state_ == DETECTING) {
            if (matched) {
                detect_count_++;
                if (detect_count_ >= tracking_thres_) {
                    detect_count_ = 0;
                    tracker_state_ = TRACKING;
                    RCLCPP_INFO(this->get_logger(), "Start tracking!");
                }
            } else {
                detect_count_ = 0;
                tracker_state_ = LOST;
                RCLCPP_WARN(this->get_logger(), "Detection failed, back to LOST.");
            }
        } else if (tracker_state_ == TRACKING) {
            if (!matched) {
                tracker_state_ = TEMP_LOST;
                lost_count_ = 1;
            }
        } else if (tracker_state_ == TEMP_LOST) {
            if (matched) {
                tracker_state_ = TRACKING;
                lost_count_ = 0;
            } else {
                lost_count_++;
                if (lost_count_ > lost_thres_) {
                    lost_count_ = 0;
                    tracker_state_ = LOST;
                    RCLCPP_INFO(this->get_logger(), "Target lost for too long, reset.");
                }
            }
        }

        // 输出滤波后的瞄准角
        // 我们使用滤波后的状态，反算当前追踪的装甲板位置，计算瞄准角
        // 如果需要打击特定面，可以在反算时加入偏移角
        Eigen::Vector3d armor_pos = ekf_.getArmorPosition(); // 当前追踪的装甲板位置
        double aim_yaw = std::atan2(armor_pos.y(), armor_pos.x());
        double aim_pitch = std::atan2(armor_pos.z(), std::sqrt(armor_pos.x()*armor_pos.x() + armor_pos.y()*armor_pos.y()));

        // 填充输出消息
        armor_interfaces::msg::ArmorArray out_msg;
        out_msg.header = msg->header;
        armor_interfaces::msg::Armor out_armor;
        // 使用匹配到的装甲板信息，或者创建一个新的
        
        // 始终发布车体中心（世界系），便于 Yolo_detect 可视化
        Eigen::Vector3d vehicle_center = ekf_.getVehiclePosition();
        out_armor.id              = tracked_id_;
        out_armor.x               = vehicle_center.x();
        out_armor.y               = vehicle_center.y();
        out_armor.z               = vehicle_center.z();
        out_armor.yaw             = ekf_.getContinuousYaw();   // 车体朝向
        out_armor.yaw_filtered    = aim_yaw;                  // 瞄准角 (yaw)
        out_armor.pitch_filtered  = aim_pitch;                // 瞄准角 (pitch)
        out_armor.is_predict      = !matched;                 // 标记是否为预测值

        out_msg.armors.push_back(out_armor);
        pub_filtered_->publish(out_msg);

        // 可视化
        // ===================== 绘制监视曲线 =====================
        if (matched) 
        {
            double raw_pos_yaw = std::atan2(matched_armor.y, matched_armor.x);
            double raw_pos_pitch = std::atan2(matched_armor.z,
                std::sqrt(matched_armor.x*matched_armor.x + matched_armor.y*matched_armor.y));
            double raw_yaw_deg   = raw_pos_yaw * 180.0 / M_PI;
            double raw_pitch_deg = raw_pos_pitch * 180.0 / M_PI;
            double filt_yaw_deg  = aim_yaw * 180.0 / M_PI;
            double filt_pitch_deg = aim_pitch * 180.0 / M_PI;
            plot_monitor_.updateAndPlotAngles(raw_yaw_deg, filt_yaw_deg,
                                            raw_pitch_deg, filt_pitch_deg);
        }

        Eigen::Vector3d pred = getPredictedArmorPosition(ekf_);
        Eigen::Vector3d obs(matched_armor.x, matched_armor.y, matched_armor.z);
        /*RCLCPP_INFO(get_logger(), "pred: [%.2f,%.2f,%.2f] obs: [%.2f,%.2f,%.2f] dist: %.3f",
                    pred.x(), pred.y(), pred.z(), obs.x(), obs.y(), obs.z(), (pred-obs).norm());*/
        //RCLCPP_INFO(this->get_logger(),"Vehicle center: [%.2f, %.2f, %.2f]", vehicle_center.x(), vehicle_center.y(), vehicle_center.z());

    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EKFNode>());
    rclcpp::shutdown();
    return 0;
}