#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <unordered_map>
#include <string>
#include "armor_interfaces/msg/armor.hpp"
#include "armor_interfaces/msg/armor_array.hpp"
#include "Kalmanfilter.hpp"

class KalmanFilterNode : public rclcpp::Node
{
public:
    KalmanFilterNode() : Node("kalman_filter_node")
    {
        RCLCPP_INFO(this->get_logger(), "卡尔曼滤波节点启动");

        sub_ = this->create_subscription<armor_interfaces::msg::ArmorArray>(
            "armor_msgs", 10,
            std::bind(&KalmanFilterNode::callback, this, std::placeholders::_1));

        // 发布滤波后的装甲板消息
        pub_ = this->create_publisher<armor_interfaces::msg::ArmorArray>(
            "armor_msgs_filtered", 10);
    }

private:
    struct TargetInfo {
        KF kf;                       // 卡尔曼滤波器
        rclcpp::Time last_time;      // 上次更新时间
        bool active;                 // 是否有效（暂时无用，可扩展）
    };

    std::unordered_map<int, TargetInfo> targets_;  // 每个 ID 的跟踪目标

    rclcpp::Subscription<armor_interfaces::msg::ArmorArray>::SharedPtr sub_;
    rclcpp::Publisher<armor_interfaces::msg::ArmorArray>::SharedPtr pub_;

    void callback(const armor_interfaces::msg::ArmorArray::SharedPtr msg)
    {
        rclcpp::Time now = msg->header.stamp;   // 使用消息中的时间戳
        armor_interfaces::msg::ArmorArray filtered_msg;
        filtered_msg.header = msg->header;      // 复制原始头（时间戳、frame_id）

        // 对每个装甲板进行处理
        for (const auto& raw : msg->armors) {
            int id = raw.id;
            auto it = targets_.find(id);

            if (it == targets_.end()) {
                // 新目标：初始化滤波器
                KF kf(0.033);   // 默认 dt，实际预测时会用动态 dt
                kf.init(raw.position.x, raw.position.y, raw.yaw);
                TargetInfo info = {kf, now, true};
                targets_[id] = info;
                // 第一帧直接复制原始数据（或直接使用原始值）
                filtered_msg.armors.push_back(raw);
                continue;
            }

            // 已存在目标：预测 + 更新
            TargetInfo& target = it->second;
            double dt = (now - target.last_time).seconds();
            if (dt > 0.0 && dt < 0.5) {   // 防止异常时间跳变
                target.kf.predict(dt);
            }
            target.kf.update(raw.position.x, raw.position.y, raw.yaw);
            target.last_time = now;

            // 获取滤波后的状态
            double fx, fy, vx, vy, fyaw;
            target.kf.getState(fx, fy, vx, vy, fyaw);

            // 构造滤波后的消息（复制原始消息，替换滤波字段）
            armor_interfaces::msg::Armor filtered;
            filtered = raw;                     // 复制角点、旋转向量、原始角度等
            filtered.position.x = fx;           // 使用滤波后的位置
            filtered.position.y = fy;
            filtered.yaw_filtered = fyaw;       // 滤波后的 yaw
            filtered_msg.armors.push_back(filtered);

            RCLCPP_INFO(this->get_logger(),"Yaw_filtered = %lf",fyaw);
        }

        // 清理长时间未更新的目标（例如超过2秒）
        auto now_clean = this->now();
        for (auto it = targets_.begin(); it != targets_.end(); ) {
            if ((now_clean - it->second.last_time).seconds() > 2.0) {
                it = targets_.erase(it);
            } else {
                ++it;
            }
        }

        pub_->publish(filtered_msg);
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