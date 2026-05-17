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
                detect_count_(0), lost_count_(0), last_yaw_(0.0), another_r_(0.26) {
        
        this->declare_parameter<int>("tracking_thres", 3);
        this->declare_parameter<int>("lost_thres", 15);
        this->declare_parameter<double>("max_match_distance", 0.6); // 严格匹配距离
        this->declare_parameter<double>("max_match_yaw_diff", 1.0); // 跳变阈值(约57度)
        
        tracking_thres_ = this->get_parameter("tracking_thres").as_int();
        lost_thres_ = this->get_parameter("lost_thres").as_int();
        max_match_distance_ = this->get_parameter("max_match_distance").as_double();
        max_match_yaw_diff_ = this->get_parameter("max_match_yaw_diff").as_double();

        sub_armor_ = this->create_subscription<armor_interfaces::msg::ArmorArray>(
            "armor_msgs", 10, std::bind(&EKFNode::armorCallback, this, _1));
        pub_filtered_ = this->create_publisher<armor_interfaces::msg::ArmorArray>(
            "armor_msgs_filtered", 10);

        plot_monitor_.initAnglePlot();
        gui_timer_ = this->create_wall_timer(10ms, [this]() { cv::waitKey(1); });

        RCLCPP_INFO(this->get_logger(), "EKF Node (10D Whole Vehicle) started.");
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
    double last_yaw_;  // 全局连续 yaw 基准
    double another_r_; 
    EKF ekf_;
    Monitor plot_monitor_;

    rclcpp::Subscription<armor_interfaces::msg::ArmorArray>::SharedPtr sub_armor_;
    rclcpp::Publisher<armor_interfaces::msg::ArmorArray>::SharedPtr pub_filtered_;
    rclcpp::TimerBase::SharedPtr gui_timer_;
    rclcpp::Time last_time_;

    // 纯数学计算，不修改全局 last_yaw_ 的状态污染！
    double computeContinuousYaw(double armor_yaw, double ref_yaw) {
        return ref_yaw + EKF::shortestAngularDistance(ref_yaw, armor_yaw);
    }

    void handleArmorJump(const armor_interfaces::msg::Armor& current_armor) {
        // 1. 处理 Yaw 的跳变（小陀螺切换面）
        double new_yaw = computeContinuousYaw(current_armor.yaw, last_yaw_);
        ekf_.x(6) = new_yaw; 
        last_yaw_ = new_yaw; // 更新基准
        
        // 2. 交换半径 r (很多机器人前后和左右装甲板的旋转半径不同)
        std::swap(ekf_.x(8), another_r_); 
        
        // ==========================================
        // 3. 完美处理“高低装甲板”的 Z 轴跳变
        // 公式：Z_armor = Zc + Zp  ==>  Zp = Z_armor - Zc
        // 逻辑：我们绝对信任当前已经收敛平稳的车体中心高度 ekf_.x(4)(Zc)，
        // 让高度差全部由 zp (ekf_.x(9)) 来吸收！
        // ==========================================
        ekf_.x(9) = current_armor.z - ekf_.x(4); 

        // 4. 为 Zp 注入不确定度，让卡尔曼滤波器在接下来几帧快速微调适应这块新板的高度
        // 【注意】千万不要增加 P(4,4) (车心的协方差)，我们要让车心高度坚如磐石！
        ekf_.P(9, 9) += 0.1; 

        RCLCPP_WARN(this->get_logger(), "Armor Jump Handled! Switched to High/Low plate. Zp updated to: %.3f", ekf_.x(9));
    }

    void armorCallback(const armor_interfaces::msg::ArmorArray::SharedPtr msg) {
        if (msg->armors.empty()) return;

        rclcpp::Time now = msg->header.stamp;
        if (last_time_.nanoseconds() == 0) { last_time_ = now; return; }
        double dt = (now - last_time_).seconds();
        last_time_ = now;
        if (dt <= 0.0 || dt > 0.5) dt = 0.01;

        if (tracker_state_ == LOST) {
            double min_dist = DBL_MAX;
            int best_idx = -1;
            for (size_t i = 0; i < msg->armors.size(); ++i) {
                double d = std::sqrt(msg->armors[i].x*msg->armors[i].x + msg->armors[i].y*msg->armors[i].y);
                if (d < min_dist) { min_dist = d; best_idx = i; }
            }
            if (best_idx >= 0) {
                const auto& armor = msg->armors[best_idx];
                tracked_id_ = armor.id;
                
                // 初始化滤波
                last_yaw_ = armor.yaw; 
                Eigen::Vector3d p_armor(armor.x, armor.y, armor.z);
                ekf_.init(p_armor, last_yaw_, 0.26, 0.0); // 初始假定 zp=0
                
                tracker_state_ = DETECTING;
                detect_count_ = 0;
            }
            return;
        }

        // --- 1. EKF 预测 ---
        ekf_.predict(dt);
        Eigen::Vector3d pred_armor_pos = ekf_.getArmorPosition();

        // --- 2. 关联同 ID 目标 ---
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
                
                // 注意：使用 computeContinuousYaw，绝对不在这里修改 last_yaw_
                double yaw_cont = computeContinuousYaw(arm.yaw, last_yaw_);
                double yaw_diff = std::abs(EKF::shortestAngularDistance(yaw_cont, ekf_.getContinuousYaw()));
                
                if (pos_diff < min_pos_diff) {
                    min_pos_diff = pos_diff;
                    best_yaw_diff = yaw_diff;
                    best_cand = i;
                }
            }

            // --- 3. 结果判断与更新 ---
            if (min_pos_diff < max_match_distance_ && best_yaw_diff < max_match_yaw_diff_) {
                matched = true;
                matched_armor = candidates[best_cand];

                // 匹配成功，真正更新 last_yaw_
                last_yaw_ = computeContinuousYaw(matched_armor.yaw, last_yaw_);

                // 准备观测值 (球面坐标与朝向)
                double obs_yaw_cam = std::atan2(matched_armor.y, matched_armor.x);
                double obs_pitch_cam = std::atan2(matched_armor.z, std::sqrt(matched_armor.x*matched_armor.x + matched_armor.y*matched_armor.y));
                double obs_dist = std::sqrt(matched_armor.x*matched_armor.x + matched_armor.y*matched_armor.y + matched_armor.z*matched_armor.z);
                
                Eigen::Vector4d z(obs_yaw_cam, obs_pitch_cam, obs_dist, last_yaw_);
                ekf_.update(z);
            } 
            else if (candidates.size() >= 1 && best_yaw_diff >= max_match_yaw_diff_) {
                // 距离相近，但角度变化巨大 -> 触发陀螺跳变处理
                handleArmorJump(candidates[best_cand]);
            }
        }

        // 状态机流转
        if (tracker_state_ == DETECTING) {
            if (matched) {
                if (++detect_count_ >= tracking_thres_) tracker_state_ = TRACKING;
            } else tracker_state_ = LOST;
        } else if (tracker_state_ == TRACKING) {
            if (!matched) { tracker_state_ = TEMP_LOST; lost_count_ = 1; }
        } else if (tracker_state_ == TEMP_LOST) {
            if (matched) { tracker_state_ = TRACKING; lost_count_ = 0; }
            else if (++lost_count_ > lost_thres_) tracker_state_ = LOST;
        }

        // --- 4. 组装与发布消息 ---
        Eigen::Vector3d curr_armor_pos = ekf_.getArmorPosition();
        double aim_yaw = std::atan2(curr_armor_pos.y(), curr_armor_pos.x());
        double aim_pitch = std::atan2(curr_armor_pos.z(), std::sqrt(curr_armor_pos.x()*curr_armor_pos.x() + curr_armor_pos.y()*curr_armor_pos.y()));

        armor_interfaces::msg::ArmorArray out_msg;
        out_msg.header = msg->header;
        
        armor_interfaces::msg::Armor out_armor;
        Eigen::Vector3d vehicle_center = ekf_.getVehiclePosition();
        
        out_armor.id              = tracked_id_;
        out_armor.x               = vehicle_center.x();
        out_armor.y               = vehicle_center.y();
        out_armor.z               = vehicle_center.z(); // 真正的车心高度
        out_armor.yaw             = ekf_.getContinuousYaw(); 
        out_armor.yaw_filtered    = aim_yaw;      // 预测击打角
        out_armor.pitch_filtered  = aim_pitch;    // 预测击打角
        out_armor.is_predict      = !matched;

        out_msg.armors.push_back(out_armor);
        pub_filtered_->publish(out_msg);

        // --- 5. 画图监控 ---
        if (matched) {
            double raw_pos_yaw = std::atan2(matched_armor.y, matched_armor.x);
            double raw_pos_pitch = std::atan2(matched_armor.z, std::sqrt(matched_armor.x*matched_armor.x + matched_armor.y*matched_armor.y));
            plot_monitor_.updateAndPlotAngles(raw_pos_yaw * 180.0 / M_PI, aim_yaw * 180.0 / M_PI,
                                              raw_pos_pitch * 180.0 / M_PI, aim_pitch * 180.0 / M_PI);
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EKFNode>());
    rclcpp::shutdown();
    return 0;
}