#include <rclcpp/rclcpp.hpp>
#include <armor_interfaces/msg/armor_array.hpp>
#include <armor_interfaces/msg/serial.hpp>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include "Kalman.hpp"
#include "Tool.hpp"

using namespace std::chrono_literals;

class KalmanFilterNode : public rclcpp::Node
{
public:
    KalmanFilterNode() : Node("kalman_filter_node")
    {
        RCLCPP_INFO(this->get_logger(), "卡尔曼滤波节点启动（绝对角度+预测输出）");

        // 预测时间（秒），可根据子弹飞行时间或系统延迟设置，例如 0.2s
        prediction_time_ = this->declare_parameter("prediction_time", 0.033);

        // 订阅装甲板检测结果（相机系三维坐标）
        sub_armor_ = this->create_subscription<armor_interfaces::msg::ArmorArray>(
            "armor_msgs", 10,
            std::bind(&KalmanFilterNode::armor_callback, this, std::placeholders::_1));

        // 订阅云台反馈角度（用于坐标变换）
        sub_serial_ = this->create_subscription<armor_interfaces::msg::Serial>(
            "serial_data", 10,
            [this](const armor_interfaces::msg::Serial::SharedPtr msg) {
                last_gimbal_yaw_ = msg->yaw;
                last_gimbal_pitch_ = msg->pitch;
            });

        // 发布滤波后的结果（绝对角度 + 预测）
        pub_filtered_ = this->create_publisher<armor_interfaces::msg::ArmorArray>(
            "armor_msgs_filtered", 10);

        // 曲线绘图（显示世界绝对角度）
        plot_tool_.initAnglePlot(200, cv::Vec2f(-40.f, 40.f), cv::Vec2f(-40.f, 40.f));
        cv::startWindowThread();
    }

    ~KalmanFilterNode() { cv::destroyAllWindows(); }

private:
    struct TargetInfo
    {
        KF kf;
        rclcpp::Time last_update_time;
        rclcpp::Time last_predict_time;
        int lost_count;
        bool initialized;
    };

    rclcpp::Subscription<armor_interfaces::msg::ArmorArray>::SharedPtr sub_armor_;
    rclcpp::Subscription<armor_interfaces::msg::Serial>::SharedPtr sub_serial_;
    rclcpp::Publisher<armor_interfaces::msg::ArmorArray>::SharedPtr pub_filtered_;
    Tool plot_tool_;

    std::unordered_map<int, TargetInfo> targets_;
    double last_gimbal_yaw_ = 0.0;
    double last_gimbal_pitch_ = 0.0;
    double prediction_time_ = 0.2;   // 预测提前量（秒）

    void armor_callback(const armor_interfaces::msg::ArmorArray::SharedPtr msg)
    {
        rclcpp::Time now = msg->header.stamp;

        // 对所有已知目标进行预测（帧间递推）
        for (auto& pair : targets_)
        {
            auto& target = pair.second;
            if (!target.initialized) continue;
            double dt = (now - target.last_predict_time).seconds();
            if (dt > 0.0 && dt < 0.5)
            {
                target.kf.predict(dt);
            }
            target.last_predict_time = now;
        }

        std::unordered_set<int> updated_ids;
        armor_interfaces::msg::ArmorArray filtered_msg;
        filtered_msg.header = msg->header;

        bool has_plot_data = false;
        double raw_yaw_world_deg = 0.0, raw_pitch_world_deg = 0.0;
        double filt_yaw_world_deg = 0.0, filt_pitch_world_deg = 0.0;

        // 处理每个检测到的装甲板
        for (const auto& raw : msg->armors)
        {
            int id = raw.id;
            updated_ids.insert(id);

            // 相机坐标系 -> 世界坐标系
            Eigen::Vector3d p_cam(raw.x, raw.y, raw.z);
            Eigen::Vector3d p_world = plot_tool_.cameraToWorld(p_cam, last_gimbal_yaw_, last_gimbal_pitch_);

            // 计算绝对角度（世界系）
            double yaw_abs_world   = std::atan2(p_world.y(), p_world.x());
            double pitch_abs_world = std::atan2(p_world.z(), std::hypot(p_world.x(), p_world.y()));

            if (!has_plot_data)
            {
                raw_yaw_world_deg   = yaw_abs_world * 180.0 / M_PI;
                raw_pitch_world_deg = pitch_abs_world * 180.0 / M_PI;
                has_plot_data = true;
            }

            auto it = targets_.find(id);
            if (it == targets_.end())
            {
                // 新目标：初始化滤波器
                KF kf;
                kf.init(yaw_abs_world, pitch_abs_world);
                TargetInfo info;
                info.kf = kf;
                info.last_update_time = now;
                info.last_predict_time = now;
                info.lost_count = 0;
                info.initialized = true;
                targets_[id] = info;

                // 第一帧无法预测，直接输出观测值
                auto filtered = raw;
                filtered.yaw_filtered   = yaw_abs_world;
                filtered.pitch_filtered = pitch_abs_world;
                filtered.is_predict = false;
                filtered_msg.armors.push_back(filtered);

                if (has_plot_data)
                {
                    filt_yaw_world_deg   = yaw_abs_world * 180.0 / M_PI;
                    filt_pitch_world_deg = pitch_abs_world * 180.0 / M_PI;
                }
            }
            else
            {
                auto& target = it->second;
                target.kf.update(yaw_abs_world, pitch_abs_world);
                target.last_update_time = now;
                target.lost_count = 0;

                double fyaw, fyaw_rate, fpitch, fpitch_rate;
                target.kf.getState(fyaw, fyaw_rate, fpitch, fpitch_rate);

                // ********** 预测未来角度（关键修改）**********
                double predicted_yaw   = fyaw + fyaw_rate * prediction_time_;
                double predicted_pitch = fpitch + fpitch_rate * prediction_time_;

                auto filtered = raw;
                filtered.yaw_filtered   = predicted_yaw;
                filtered.pitch_filtered = predicted_pitch;
                filtered.is_predict = false;   // 虽然是预测值，但可以标记为 false，由电控决定是否视为预测
                filtered_msg.armors.push_back(filtered);

                if (has_plot_data)
                {
                    filt_yaw_world_deg   = fyaw * 180.0 / M_PI;      // 显示滤波后的当前角度（不带预测）
                    filt_pitch_world_deg = fpitch * 180.0 / M_PI;
                }
            }
        }

        // 处理丢失的目标（同样输出预测后的绝对角度）
        for (auto& pair : targets_)
        {
            int id = pair.first;
            auto& target = pair.second;
            if (!target.initialized) continue;
            if (updated_ids.find(id) != updated_ids.end()) continue;

            target.lost_count++;
            if (target.lost_count > 10) continue;

            double fyaw, fyaw_rate, fpitch, fpitch_rate;
            target.kf.getState(fyaw, fyaw_rate, fpitch, fpitch_rate);

            double predicted_yaw   = fyaw + fyaw_rate * prediction_time_;
            double predicted_pitch = fpitch + fpitch_rate * prediction_time_;

            armor_interfaces::msg::Armor predicted;
            predicted.id = id;
            predicted.yaw_filtered   = predicted_yaw;
            predicted.pitch_filtered = predicted_pitch;
            predicted.is_predict = true;   // 标记为预测值
            filtered_msg.armors.push_back(predicted);
        }

        // 清理长时间未更新的目标
        auto now_clean = this->now();
        for (auto it = targets_.begin(); it != targets_.end(); )
        {
            if ((now_clean - it->second.last_update_time).seconds() > 2.0)
                it = targets_.erase(it);
            else
                ++it;
        }

        pub_filtered_->publish(filtered_msg);

        // 绘图（显示当前滤波值，不显示预测值，便于观察）
        if (has_plot_data)
        {
            plot_tool_.updateAndPlotAngles(raw_yaw_world_deg, filt_yaw_world_deg,
                                           raw_pitch_world_deg, filt_pitch_world_deg);
        }
    }
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<KalmanFilterNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}