#include <rclcpp/rclcpp.hpp>
#include <eigen3/Eigen/Dense>
#include "armor_interfaces/msg/armor_array.hpp"
#include "armor_interfaces/msg/serial.hpp"
#include "EKF.hpp"
#include "Tool.hpp"

class EKFNode : public rclcpp::Node {
public:
    EKFNode() : Node("ekf_node") {
        // 订阅检测到的装甲板（相机系坐标，mm；yaw/pitch 为绝对角度 rad）
        sub_armor_ = this->create_subscription<armor_interfaces::msg::ArmorArray>(
            "armor_msgs", 10, std::bind(&EKFNode::armorCallback, this, std::placeholders::_1));
        
        // 订阅云台实时角度（用于坐标变换）
        sub_serial_ = this->create_subscription<armor_interfaces::msg::Serial>(
            "serial_data", 10, [this](const armor_interfaces::msg::Serial::SharedPtr msg) {
                last_gimbal_yaw_ = msg->yaw;   // rad
                last_gimbal_pitch_ = msg->pitch;
            });

        pub_filtered_ = this->create_publisher<armor_interfaces::msg::ArmorArray>("armor_msgs_filtered", 10);

        // ---------- 曲线绘制初始化 ----------
       plot_tool_.initAnglePlot(200, cv::Vec2f(-40.f, 40.f), cv::Vec2f(-40.f, 40.f));
        cv::startWindowThread();        // 独立线程刷新窗口，避免手动 waitKey

        RCLCPP_INFO(this->get_logger(), "EKF Node Started.");
    }

private:
    void armorCallback(const armor_interfaces::msg::ArmorArray::SharedPtr msg) 
    {
        if (msg->armors.empty()) return;

        // 1. 计算时间步长 dt
        rclcpp::Time now = msg->header.stamp;
        if (last_time_.nanoseconds() == 0) { last_time_ = now; return; }
        double dt = (now - last_time_).seconds();
        last_time_ = now;
        if (dt <= 0 || dt > 0.1) dt = 0.033; // 异常处理

        armor_interfaces::msg::ArmorArray out_msg;
        out_msg.header = msg->header;

        bool has_plot_data = false;
        double raw_y = 0, filt_y = 0;
        double raw_p = 0, filt_p = 0;

        for (const auto& armor : msg->armors) {
            // 2. 坐标转换
            Eigen::Vector3d p_cam(armor.x, armor.y, armor.z);
            Eigen::Vector3d p_world = plot_tool_.cameraToWorld(p_cam, last_gimbal_yaw_, last_gimbal_pitch_);
            
            double abs_yaw = armor.yaw;
            double abs_pitch = armor.pitch; // 新增：提取观测俯仰角

            // 3. EKF 逻辑
            if (ekf_map_.find(armor.id) == ekf_map_.end()) {
                ekf_map_[armor.id].init(p_world, abs_yaw, abs_pitch);
            }

            auto& ekf = ekf_map_[armor.id];
            ekf.predict(dt);
            ekf.update(p_world, abs_yaw, abs_pitch);

            // 4. 获取结果
            Eigen::Vector3d pos_c;
            double aim_yaw, v_yaw, aim_pitch, v_pitch;
            ekf.getState(pos_c, aim_yaw, v_yaw, aim_pitch, v_pitch);

            // 5. 填入输出消息
            auto out_armor = armor;
            out_armor.yaw_filtered = aim_yaw;
            out_armor.pitch_filtered = aim_pitch; // 新增：填入滤波后的 Pitch
            out_msg.armors.push_back(out_armor);

            // 6. 捕捉第一块装甲板用于可视化绘图
            if (!has_plot_data) {
                raw_y = abs_yaw * 180.0 / M_PI;
                filt_y = aim_yaw * 180.0 / M_PI;
                raw_p = abs_pitch * 180.0 / M_PI;
                filt_p = aim_pitch * 180.0 / M_PI;
                has_plot_data = true;
            }
        }

        pub_filtered_->publish(out_msg);

        // 7. 只有数据有效时才绘图，且保证每帧刷新
        if (has_plot_data) {
            RCLCPP_INFO(this->get_logger(), "画图数值 -> Raw Pitch: %.2f, Filt Pitch: %.2f", raw_p, filt_p);
            plot_tool_.updateAndPlotAngles(raw_y, filt_y, raw_p, filt_p); 
        }
    }

    // 成员变量
    std::unordered_map<int, EKF> ekf_map_;
    double last_gimbal_yaw_ = 0.0, last_gimbal_pitch_ = 0.0;
    rclcpp::Time last_time_;

    rclcpp::Subscription<armor_interfaces::msg::ArmorArray>::SharedPtr sub_armor_;
    rclcpp::Subscription<armor_interfaces::msg::Serial>::SharedPtr sub_serial_;
    rclcpp::Publisher<armor_interfaces::msg::ArmorArray>::SharedPtr pub_filtered_;

    Tool plot_tool_;   // 曲线绘制工具
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EKFNode>());
    rclcpp::shutdown();
    return 0;
}