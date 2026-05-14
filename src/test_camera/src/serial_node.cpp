#include "rclcpp/rclcpp.hpp"
#include "armor_interfaces/msg/armor_array.hpp"
#include "armor_interfaces/msg/serial.hpp"
#include "serial_driver/serial_driver.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

// ============ 新协议常量 ============
#define VISION_TO_CTRL_HEAD    0xFE
#define VISION_TO_CTRL_TAIL    0xFF
#define VISION_TO_CTRL_LEN     11      // 帧头(1) + yaw(4) + pitch(4) + valid(1) + 帧尾(1)

#define CTRL_TO_VISION_HEAD    0xFE
#define CTRL_TO_VISION_TAIL    0xFF
#define CTRL_TO_VISION_LEN     10      // 帧头(1) + yaw(4) + pitch(4) + 帧尾(1)
// ===================================

class SerialNode : public rclcpp::Node
{
public:
    SerialNode() : Node("serial_node"), is_running_(true)
    {
        // 声明参数（可在启动时覆盖）
        this->declare_parameter<std::string>("port", "/dev/ttyACM0");
        this->declare_parameter<int>("baud", 115200);

        std::string port = this->get_parameter("port").as_string();
        int baud = this->get_parameter("baud").as_int();

        RCLCPP_INFO(this->get_logger(), "SerialNode 启动，端口: %s, 波特率: %d", port.c_str(), baud);

        // 初始化串口
        try
        {
            owned_ctx_ = std::make_unique<IoContext>(2);
            serial_driver_ = std::make_unique<drivers::serial_driver::SerialDriver>(*owned_ctx_);

            drivers::serial_driver::SerialPortConfig config(
                static_cast<uint32_t>(baud),
                drivers::serial_driver::FlowControl::NONE,
                drivers::serial_driver::Parity::NONE,
                drivers::serial_driver::StopBits::ONE
            );

            serial_driver_->init_port(port, config);
            if (!serial_driver_->port()->is_open())
                serial_driver_->port()->open();

            RCLCPP_INFO(this->get_logger(), "串口 %s 打开成功", port.c_str());
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "串口打开失败：%s", e.what());
            return;
        }

        // 启动接收线程（确保串口打开成功后再启动）
        receive_thread_ = std::thread(&SerialNode::receive_loop, this);

        // 订阅卡尔曼滤波后的绝对角度
        sub_ = this->create_subscription<armor_interfaces::msg::ArmorArray>(
            "armor_msgs_filtered", 10,
            std::bind(&SerialNode::armor_callback, this, std::placeholders::_1));

        // 发布从电控接收到的云台角度
        pub_ = this->create_publisher<armor_interfaces::msg::Serial>("serial_data", 10);
    }

    ~SerialNode()
    {
        is_running_ = false;
        if (receive_thread_.joinable())
            receive_thread_.join();
        if (serial_driver_ && serial_driver_->port()->is_open())
            serial_driver_->port()->close();
        if (owned_ctx_)
            owned_ctx_->waitForExit();
    }

private:
    rclcpp::Subscription<armor_interfaces::msg::ArmorArray>::SharedPtr sub_;
    rclcpp::Publisher<armor_interfaces::msg::Serial>::SharedPtr pub_;

    std::unique_ptr<IoContext> owned_ctx_;
    std::unique_ptr<drivers::serial_driver::SerialDriver> serial_driver_;
    std::thread receive_thread_;
    std::atomic<bool> is_running_;

    std::atomic<float> latest_gimbal_yaw_{0.0f};
    std::atomic<float> latest_gimbal_pitch_{0.0f};

    // 发送自瞄指令（视觉→电控，11字节）
    void armor_callback(armor_interfaces::msg::ArmorArray::SharedPtr msg)
    {
        if (!serial_driver_ || !serial_driver_->port()->is_open())
            return;

        uint8_t buf[VISION_TO_CTRL_LEN] = {0};
        buf[0] = VISION_TO_CTRL_HEAD;          // 帧头
        buf[9] = 0;                            // target_valid 默认为 0（无效）
        buf[10] = VISION_TO_CTRL_TAIL;         // 帧尾

        if (!msg->armors.empty())
        {
            const auto& armor = msg->armors[0];

            // 弧度转度
            float yaw_deg   = static_cast<float>(armor.yaw_filtered   * 180.0 / M_PI);
            float pitch_deg = static_cast<float>(armor.pitch_filtered * 180.0 / M_PI);

            memcpy(&buf[1], &yaw_deg, 4);      // yaw
            memcpy(&buf[5], &pitch_deg, 4);    // pitch
            buf[9] = 1;                        // target_valid = 1 有效

            std::vector<uint8_t> send_data(buf, buf + VISION_TO_CTRL_LEN);
            serial_driver_->port()->send(send_data);

            RCLCPP_INFO(this->get_logger(), "发送自瞄指令: yaw=%.2f°, pitch=%.2f°", yaw_deg, pitch_deg);
        }
        else
        {
            // 无目标时也发送一帧（target_valid=0），保持心跳
            std::vector<uint8_t> send_data(buf, buf + VISION_TO_CTRL_LEN);
            serial_driver_->port()->send(send_data);
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                 "未识别到装甲板，发送无效帧");
        }
    }

    // 接收线程（解析电控→视觉，10字节）
    void receive_loop()
    {
        std::vector<uint8_t> ring_buffer;

        while (is_running_ && rclcpp::ok())
        {
            if (!serial_driver_ || !serial_driver_->port()->is_open())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            try
            {
                std::vector<uint8_t> data;
                serial_driver_->port()->receive(data);   // 阻塞读取

                if (!data.empty())
                {
                    ring_buffer.insert(ring_buffer.end(), data.begin(), data.end());

                    // 帧同步：查找帧头 0xFE
                    while (ring_buffer.size() >= CTRL_TO_VISION_LEN)
                    {
                        auto it = std::find(ring_buffer.begin(), ring_buffer.end(), CTRL_TO_VISION_HEAD);
                        if (it == ring_buffer.end())
                        {
                            ring_buffer.clear();
                            break;
                        }
                        ring_buffer.erase(ring_buffer.begin(), it);

                        if (ring_buffer.size() < CTRL_TO_VISION_LEN)
                            break;

                        // 取出一帧
                        std::vector<uint8_t> frame(ring_buffer.begin(),
                                                   ring_buffer.begin() + CTRL_TO_VISION_LEN);

                        // 校验帧尾
                        if (frame[CTRL_TO_VISION_LEN - 1] == CTRL_TO_VISION_TAIL)
                        {
                            float yaw_deg, pitch_deg;
                            memcpy(&yaw_deg,   &frame[1], 4);
                            memcpy(&pitch_deg, &frame[5], 4);

                            // 转换为弧度存储/发布
                            constexpr float DEG2RAD = static_cast<float>(M_PI / 180.0);
                            float yaw_rad   = yaw_deg   * DEG2RAD;
                            float pitch_rad = pitch_deg * DEG2RAD;

                            latest_gimbal_yaw_   = yaw_rad;
                            latest_gimbal_pitch_ = pitch_rad;

                            auto msg = armor_interfaces::msg::Serial();
                            msg.header.stamp = this->now();
                            msg.yaw   = yaw_rad;
                            msg.pitch = pitch_rad;
                            pub_->publish(msg);

                            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                                                 "收到云台角度: yaw=%.2f°, pitch=%.2f°",
                                                 yaw_deg, pitch_deg);
                        }
                        else
                        {
                            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                                 "帧尾错误: 期望 0x%02X, 实际 0x%02X",
                                                 CTRL_TO_VISION_TAIL, frame[CTRL_TO_VISION_LEN - 1]);
                        }

                        // 从缓冲区移除已处理帧
                        ring_buffer.erase(ring_buffer.begin(), ring_buffer.begin() + CTRL_TO_VISION_LEN);
                    }
                }
            }
            catch (const std::exception& e)
            {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                     "接收异常: %s", e.what());
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SerialNode>());
    rclcpp::shutdown();
    return 0;
}