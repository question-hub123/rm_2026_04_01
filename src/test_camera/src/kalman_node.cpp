#include <rclcpp/rclcpp.hpp>
#include <eigen3/Eigen/Dense>
#include "armor_interfaces/msg/armor_array.hpp"
#include "armor_interfaces/msg/serial.hpp"
#include "AngleKalman.hpp"
#include "Tool.hpp"

class EKFNode : public rclcpp::Node {
public:
    EKFNode() : Node("ekf_node") {
        sub_armor_ = this->create_subscription<armor_interfaces::msg::ArmorArray>(
            "armor_msgs", 10, std::bind(&EKFNode::armorCallback, this, std::placeholders::_1));

        sub_serial_ = this->create_subscription<armor_interfaces::msg::Serial>(
            "serial_data", 10, [this](const armor_interfaces::msg::Serial::SharedPtr msg) {
                last_gimbal_yaw_ = msg->yaw;
                last_gimbal_pitch_ = msg->pitch;
            });

        pub_filtered_ = this->create_publisher<armor_interfaces::msg::ArmorArray>(
            "armor_msgs_filtered", 10);

        plot_tool_.initAnglePlot(200, cv::Vec2f(-30.f, 30.f), cv::Vec2f(-10.f, 10.f));
        cv::startWindowThread();
        RCLCPP_INFO(this->get_logger(), "EKF Node Started (12-dim CA model).");
    }

private:
    Eigen::Vector3d cameraToWorld(const Eigen::Vector3d& p_cam) {
        return plot_tool_.cameraToWorld(p_cam, last_gimbal_yaw_, last_gimbal_pitch_);
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
        if (dt <= 0.0 || dt > 0.5) dt = 0.01;

        armor_interfaces::msg::ArmorArray out_msg;
        out_msg.header = msg->header;

        bool plot_data_valid = false;
        double raw_yaw_deg = 0.0, raw_pitch_deg = 0.0;
        double filt_yaw_deg = 0.0, filt_pitch_deg = 0.0;

        for (const auto& armor : msg->armors) {
            Eigen::Vector3d p_cam(armor.x / 1000.0, armor.y / 1000.0, armor.z / 1000.0);
            Eigen::Vector3d p_world = cameraToWorld(p_cam);

            double abs_yaw   = armor.yaw;
            double abs_pitch = armor.pitch;

            auto it = ekf_map_.find(armor.id);
            if (it == ekf_map_.end()) {
                EKF ekf;
                ekf.init(p_world, abs_yaw);
                ekf_map_.emplace(armor.id, ekf);
                continue;
            }

            auto& ekf = it->second;
            ekf.predict(dt);
            ekf.update(p_world, abs_yaw);

            Eigen::Vector3d pos_c;
            double yaw_c, v_yaw;
            ekf.getState(pos_c, yaw_c, v_yaw);

            double predict_t = 0.05;
            double pred_x = pos_c.x() + ekf.getVx() * predict_t + 0.5 * ekf.getAx() * predict_t * predict_t;
            double pred_y = pos_c.y() + ekf.getVy() * predict_t + 0.5 * ekf.getAy() * predict_t * predict_t;
            double pred_z = pos_c.z() + ekf.getVz() * predict_t + 0.5 * ekf.getAz() * predict_t * predict_t;
            double pred_yaw = yaw_c + v_yaw * predict_t + 0.5 * ekf.getAYaw() * predict_t * predict_t;

            const double R = 0.25;
            double armor_x = pred_x + R * std::sin(pred_yaw);
            double armor_z = pred_z + R * std::cos(pred_yaw);
            double aim_yaw   = std::atan2(armor_x, armor_z);
            double aim_pitch = std::atan2(pred_y, std::sqrt(armor_x*armor_x + armor_z*armor_z));

            armor_interfaces::msg::Armor out_armor = armor;
            out_armor.yaw = aim_yaw;
            out_armor.pitch = aim_pitch;
            out_armor.yaw_filtered = aim_yaw;
            out_armor.pitch_filtered = aim_pitch;
            out_armor.is_predict = false;
            out_msg.armors.push_back(out_armor);

            if (!plot_data_valid) {
                raw_yaw_deg   = abs_yaw   * 180.0 / M_PI;
                raw_pitch_deg = abs_pitch * 180.0 / M_PI;
                filt_yaw_deg   = aim_yaw   * 180.0 / M_PI;
                filt_pitch_deg = aim_pitch * 180.0 / M_PI;
                plot_data_valid = true;
            }
        }

        pub_filtered_->publish(out_msg);

        if (plot_data_valid) {
            plot_tool_.updateAndPlotAngles(raw_yaw_deg, filt_yaw_deg,
                                           raw_pitch_deg, filt_pitch_deg);
        }
    }

    std::unordered_map<int, EKF> ekf_map_;
    double last_gimbal_yaw_ = 0.0, last_gimbal_pitch_ = 0.0;
    rclcpp::Time last_time_;

    rclcpp::Subscription<armor_interfaces::msg::ArmorArray>::SharedPtr sub_armor_;
    rclcpp::Subscription<armor_interfaces::msg::Serial>::SharedPtr sub_serial_;
    rclcpp::Publisher<armor_interfaces::msg::ArmorArray>::SharedPtr pub_filtered_;

    Tool plot_tool_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EKFNode>());
    rclcpp::shutdown();
    return 0;
}