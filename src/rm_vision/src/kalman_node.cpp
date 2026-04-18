#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <unordered_map>
#include <string>
#include <unordered_set>
#include "armor_interfaces/msg/armor.hpp"
#include "armor_interfaces/msg/armor_array.hpp"
#include "Kalman.hpp"

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
        rclcpp::Time last_update_time;      // 上次更新时间
        rclcpp::Time last_predict_time;
        int lost_count;
        double last_yaw;
        double last_pitch;
        double last_yaw_rate;
        double last_pitch_rate;
        bool initialized;
    };

    std::unordered_map<int, TargetInfo> targets_;  // 每个 ID 的跟踪目标

    rclcpp::Subscription<armor_interfaces::msg::ArmorArray>::SharedPtr sub_;
    rclcpp::Publisher<armor_interfaces::msg::ArmorArray>::SharedPtr pub_;

    void callback(const armor_interfaces::msg::ArmorArray::SharedPtr msg)
    {
        rclcpp::Time now = msg->header.stamp;   // 使用消息中的时间戳
        armor_interfaces::msg::ArmorArray filtered_msg;
        filtered_msg.header = msg->header;      // 复制原始头（时间戳、frame_id）

        for(auto& pair : targets_)
        {
            auto& target = pair.second;
            if(!target.initialized) continue;
            double dt = (now - target.last_predict_time).seconds();
            if(dt > 0.0 && dt < 0.5) target.kf.predict(dt);

            target.last_predict_time = now;
        }

        std::unordered_set<int> updated_ids;
        for(const auto& raw : msg->armors)
        {
            int id = raw.id;
            updated_ids.insert(id);
            auto it = targets_.find(id);

            if(it == targets_.end())
            {
                KF kf;
                kf.init(raw.yaw, raw.pitch);
                TargetInfo info;
                info.kf = kf;
                info.last_update_time = now;
                info.last_predict_time = now;
                info.lost_count = 0;
                info.initialized = true;
                info.last_pitch = raw.pitch;
                info.last_yaw = raw.yaw;
                targets_[id] = info;

                auto filtered = raw;
                filtered.yaw_filtered = raw.yaw;
                filtered.pitch_filtered = raw.pitch;
                filtered.is_predict = false;
                filtered_msg.armors.push_back(filtered);
            }
            else
            {
                auto& target = it->second;
                target.kf.update(raw.yaw, raw.pitch);
                target.last_update_time = now;
                target.lost_count = 0;

                // 获取滤波后的状态
                double fyaw, fyaw_rate, fpitch, fpitch_rate;
                target.kf.getState(fyaw, fyaw_rate, fpitch, fpitch_rate);
                target.last_yaw = fyaw;
                target.last_pitch = fpitch;

                auto filtered = raw;
                filtered.yaw_filtered = fyaw;
                filtered.pitch_filtered = fpitch;
                filtered.is_predict = false;
                filtered_msg.armors.push_back(filtered);
            }
        }

        for (auto& pair : targets_) 
        {
            int id = pair.first;
            auto& target = pair.second;
            if (!target.initialized) continue;

            // 如果本次没有被更新，则认为是丢失
            if (updated_ids.find(id) == updated_ids.end()) {
                target.lost_count++;
                if (target.lost_count > 10) {   // 连续丢失超过10帧，不再发布
                    continue;
                }

                // 获取当前预测状态
                double fyaw, fyaw_rate, fpitch, fpitch_rate;
                target.kf.getState(fyaw, fyaw_rate, fpitch, fpitch_rate);

                armor_interfaces::msg::Armor predicted;
                predicted.id = id;
                predicted.yaw = fyaw;
                predicted.pitch = fpitch;
                predicted.yaw_filtered = fyaw;
                predicted.pitch_filtered = fpitch;
                predicted.is_predict = true;    // ★ 标记为预测值
                // 其他字段（位置、角点）无法预测，保持默认

                filtered_msg.armors.push_back(predicted);
            }
        }

        // 第四步：清理长时间未更新的目标（超过2秒）
        auto now_clean = this->now();
        for (auto it = targets_.begin(); it != targets_.end(); ) {
            if ((now_clean - it->second.last_update_time).seconds() > 2.0) {
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