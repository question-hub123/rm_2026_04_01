#include <rclcpp/rclcpp.hpp>
#include <eigen3/Eigen/Dense>
#include "armor_interfaces/msg/armor_array.hpp"
#include "armor_interfaces/msg/serial.hpp"
#include "EKF.hpp"
#include "Tool.hpp"

using namespace std::chrono_literals;

class EKFNode : public rclcpp::Node {
public:
    EKFNode() : Node("ekf_node") {
        sub_armor_ = this->create_subscription<armor_interfaces::msg::ArmorArray>(
            "armor_msgs", 10, std::bind(&EKFNode::armorCallback, this, std::placeholders::_1));

        sub_serial_ = this->create_subscription<armor_interfaces::msg::Serial>(
            "serial_data", 10, [this](const armor_interfaces::msg::Serial::SharedPtr msg) {
                last_gimbal_yaw_   = msg->yaw;    // rad
                last_gimbal_pitch_ = msg->pitch;  // rad
            });

        pub_filtered_ = this->create_publisher<armor_interfaces::msg::ArmorArray>(
            "armor_msgs_filtered", 10);

        // 初始化角度曲线窗口
        plot_tool_.initAnglePlot(200, cv::Vec2f(-40.f, 40.f), cv::Vec2f(-40.f, 40.f));

        // 定时器驱动 GUI 事件循环（10ms 一次），防止 OpenCV 报错
        gui_timer_ = this->create_wall_timer(10ms, [this]() { cv::waitKey(1); });

        RCLCPP_INFO(this->get_logger(), "EKF Node Started (13-dim CA model with online radius).");
    }

private:
    // ------------------------------------------------------------
    // 坐标变换：相机系 → 世界系（利用云台当前角度）
    // ------------------------------------------------------------
    void cameraToWorld(const Eigen::Vector3d& p_cam, double yaw_cam,
                       Eigen::Vector3d& p_world, double& yaw_world) 
    {
        // OpenCV 相机系 → 云台初始系（x前, y左, z上）
        Eigen::Vector3d p_gimbal_init;
        p_gimbal_init << p_cam.z(), -p_cam.x(), -p_cam.y();

        // 云台旋转（先 pitch 后 yaw）
        Eigen::AngleAxisd pitch_rot(last_gimbal_pitch_, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yaw_rot(last_gimbal_yaw_,   Eigen::Vector3d::UnitZ());
        Eigen::Matrix3d R_world = yaw_rot.toRotationMatrix() * pitch_rot.toRotationMatrix();

        p_world = R_world * p_gimbal_init;

        // 装甲板法向量变换
        Eigen::AngleAxisd yaw_cam_rot(yaw_cam, Eigen::Vector3d::UnitY());
        Eigen::Vector3d normal_cam = yaw_cam_rot * Eigen::Vector3d(0.0, 0.0, 1.0);
        Eigen::Vector3d normal_gimbal_init;
        normal_gimbal_init << normal_cam.z(), -normal_cam.x(), -normal_cam.y();
        Eigen::Vector3d normal_world = R_world * normal_gimbal_init;

        yaw_world = std::atan2(normal_world.x(), normal_world.z());
    }

    // 只转换位置（不需要朝向时使用）
    Eigen::Vector3d cameraToWorld(const Eigen::Vector3d& p_cam) {
        Eigen::Vector3d p_world;
        double dummy;
        cameraToWorld(p_cam, 0.0, p_world, dummy);
        return p_world;
    }

    // ------------------------------------------------------------
    // 装甲板检测回调
    // ------------------------------------------------------------
    void armorCallback(const armor_interfaces::msg::ArmorArray::SharedPtr msg) {
        if (msg->armors.empty()) return;

        rclcpp::Time now = msg->header.stamp;
        if (last_time_.nanoseconds() == 0) {
            last_time_ = now;
            return;
        }
        double dt = (now - last_time_).seconds();
        last_time_ = now;
        if (dt <= 0.0 || dt > 0.5) dt = 0.01;   // 异常保护

        armor_interfaces::msg::ArmorArray out_msg;
        out_msg.header = msg->header;

        bool plot_data_valid = false;
        double raw_yaw_deg = 0.0, raw_pitch_deg = 0.0;
        double filt_yaw_deg = 0.0, filt_pitch_deg = 0.0;

        for (const auto& armor : msg->armors) {
            // 单位转换 mm → m
            Eigen::Vector3d p_cam(armor.x / 1000.0, armor.y / 1000.0, armor.z / 1000.0);

            double yaw_cam = armor.yaw;      // 相机系下装甲板朝向
            double pitch_cam = armor.pitch;  // 相机系下俯仰角

            // 转换到世界系
            Eigen::Vector3d p_world;
            double yaw_world;
            cameraToWorld(p_cam, yaw_cam, p_world, yaw_world);

            // 查找或初始化该目标的 EKF
            auto it = ekf_map_.find(armor.id);
            if (it == ekf_map_.end()) {
                EKF ekf;
                ekf.init(p_world, yaw_world);
                ekf_map_.emplace(armor.id, ekf);
                continue;   // 第一帧不输出
            }

            // 预测 + 更新
            auto& ekf = it->second;
            ekf.predict(dt);
            ekf.update(p_world, yaw_world);

            // 获取滤波后的车体状态
            Eigen::Vector3d pos_c;
            double yaw_c, v_yaw, r_est;
            ekf.getState(pos_c, yaw_c, v_yaw, r_est);

            // 如需加速度信息，取消下面注释（并确保 EKF 类中已实现对应函数）
            // double ax_c = ekf.getAx();
            // double ay_c = ekf.getAy();
            // double az_c = ekf.getAz();
            // double a_yaw = ekf.getAYaw();

            // 预测未来 50ms 的车体状态（延迟补偿）
            double predict_t = 0.05;
            double pred_x = pos_c.x() + ekf.getVx() * predict_t; // +0.5*ax_c*predict_t^2
            double pred_y = pos_c.y() + ekf.getVy() * predict_t;
            double pred_z = pos_c.z() + ekf.getVz() * predict_t;
            double pred_yaw = yaw_c + v_yaw * predict_t;

            // 反推装甲板位置，计算绝对瞄准角
            double armor_x = pred_x + r_est * std::sin(pred_yaw);
            double armor_z = pred_z + r_est * std::cos(pred_yaw);
            double aim_yaw   = std::atan2(armor_x, armor_z);   // 绝对 yaw (rad)
            double aim_pitch = std::atan2(pred_y, std::sqrt(armor_x*armor_x + armor_z*armor_z));

            // 填充输出消息
            armor_interfaces::msg::Armor out_armor = armor;
            out_armor.yaw = aim_yaw;
            out_armor.pitch = aim_pitch;
            out_armor.yaw_filtered = aim_yaw;
            out_armor.pitch_filtered = aim_pitch;
            out_armor.is_predict = false;
            out_msg.armors.push_back(out_armor);

            // 为绘图收集数据（取第一个有效目标）
            if (!plot_data_valid) {
                // 原始观测（世界系）
                raw_yaw_deg   = yaw_world * 180.0 / M_PI;
                // 原始俯仰角：基于装甲板世界坐标计算
                double raw_pitch_world = std::atan2(p_world.z(), std::hypot(p_world.x(), p_world.y()));
                raw_pitch_deg = raw_pitch_world * 180.0 / M_PI;

                // 滤波后的瞄准角（用于曲线显示）
                filt_yaw_deg   = aim_yaw * 180.0 / M_PI;
                filt_pitch_deg = aim_pitch * 180.0 / M_PI;
                plot_data_valid = true;
            }
        }

        pub_filtered_->publish(out_msg);

        // 更新角度曲线图
        if (plot_data_valid) {
            plot_tool_.updateAndPlotAngles(raw_yaw_deg, filt_yaw_deg,
                                           raw_pitch_deg, filt_pitch_deg);
        }
    }

    // 成员变量
    std::unordered_map<int, EKF> ekf_map_;
    double last_gimbal_yaw_ = 0.0;
    double last_gimbal_pitch_ = 0.0;
    rclcpp::Time last_time_;
    Tool plot_tool_;

    rclcpp::Subscription<armor_interfaces::msg::ArmorArray>::SharedPtr sub_armor_;
    rclcpp::Subscription<armor_interfaces::msg::Serial>::SharedPtr sub_serial_;
    rclcpp::Publisher<armor_interfaces::msg::ArmorArray>::SharedPtr pub_filtered_;
    rclcpp::TimerBase::SharedPtr gui_timer_;   // GUI 驱动定时器
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EKFNode>());
    rclcpp::shutdown();
    return 0;
}