#include "rclcpp/rclcpp.hpp"
#include "armor_interfaces/msg/armor_array.hpp"
#include "armor_interfaces/msg/serial.hpp"    // 自定义消息，包含 yaw/pitch
#include "serial_driver/serial_driver.hpp"
#include <armor_interfaces/msg/detail/serial__struct.hpp>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>

#define TX_FRAME_LEN 11
#define HEAD 0x0A5
#define TAIL 0x01

class SerialNode : public rclcpp::Node
{
public:
    SerialNode() : Node("serial_node"), is_running_(true)
    {
        RCLCPP_INFO(this->get_logger(), "SerialNode 启动");

        const std::string port = "/dev/ttyUSB0";
        const int baud = 115200;

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
            {
                serial_driver_->port()->open();
            }
            
            RCLCPP_INFO(this->get_logger(), "串口 %s 打开成功，波特率 %d", port.c_str(), baud);
        } 
        catch (const std::exception& e) 
        {
            RCLCPP_ERROR(this->get_logger(), "串口打开失败：%s", e.what());
            return;
        }

        // 启动接收线程
        receive_thread_ = std::thread(&SerialNode::receive_loop, this);

        // 订阅视觉节点发布的装甲板信息，用于向电控发送自瞄指令
        sub_ = this->create_subscription<armor_interfaces::msg::ArmorArray>(
            "armor_msgs_filtered", 10,
            std::bind(&SerialNode::callback, this, std::placeholders::_1)
        );

        // 发布从电控接收到的云台角度（供视觉节点使用）
        pub_ = this->create_publisher<armor_interfaces::msg::Serial>("serial_data", 10);
    }

    ~SerialNode()
    {
        is_running_ = false;
        if (receive_thread_.joinable()) {
            receive_thread_.join();
        }
        if (serial_driver_ && serial_driver_->port()->is_open()) {
            serial_driver_->port()->close();
        }
        if (owned_ctx_) {
            owned_ctx_->waitForExit();
        }
    }

    // 供外部获取最新云台角度（原子操作，线程安全）
    float getGimbalYaw() const { return latest_gimbal_yaw_.load(); }
    float getGimbalPitch() const { return latest_gimbal_pitch_.load(); }

private:
    // 订阅视觉节点
    rclcpp::Subscription<armor_interfaces::msg::ArmorArray>::SharedPtr sub_;
    // 发布云台角度
    rclcpp::Publisher<armor_interfaces::msg::Serial>::SharedPtr pub_;
    
    std::unique_ptr<IoContext> owned_ctx_;
    std::unique_ptr<drivers::serial_driver::SerialDriver> serial_driver_;
    std::thread receive_thread_;
    std::atomic<bool> is_running_;

    // 接收到的云台角度（单位：度）
    std::atomic<float> latest_gimbal_yaw_{0.0f};
    std::atomic<float> latest_gimbal_pitch_{0.0f};

    // 视觉回调：将自瞄指令发送给电控
    void callback(armor_interfaces::msg::ArmorArray::SharedPtr msg)
    {
        if (!serial_driver_ || !serial_driver_->port()->is_open()) return;

        if (msg->armors.empty())
        {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "未识别到装甲板");
            return;
        }

        // 简单策略：取第一个装甲板
        const auto& armor = msg->armors[0];

        // 注意：视觉节点发送的 yaw/pitch 已经是绝对角度（度），直接使用
        float yaw_deg = armor.yaw_filtered * 180.0 / M_PI;      // 已滤波的绝对偏航角
        float pitch_deg = armor.pitch_filtered * 180.0 / M_PI;  // 已滤波的绝对俯仰角

        uint8_t buf[TX_FRAME_LEN] = {0};
        buf[0] = HEAD;
        memcpy(&buf[1], &pitch_deg, 4);
        memcpy(&buf[5], &yaw_deg, 4);
        buf[9] = TAIL;

        //bool tes = armor.is_predict;

        std::vector<uint8_t> send_data(buf, buf + TX_FRAME_LEN);
        serial_driver_->port()->send(send_data);

        RCLCPP_INFO(this->get_logger(), "发送自瞄指令: yaw=%.2f°, pitch=%.2f°", yaw_deg, pitch_deg);
    }

    // 接收线程：持续读取串口数据，解析云台角度并发布
    void receive_loop()
    {
        while (is_running_ && rclcpp::ok())
        {
            if (!serial_driver_ || !serial_driver_->port()->is_open()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            try 
            {
                std::vector<uint8_t> data;
                serial_driver_->port()->receive(data);
                
                if (!data.empty())
                {
                    // 简单协议：帧头 + 4字节pitch + 4字节yaw + 帧尾
                    if (data.size() >= 10 && data[0] == HEAD && data.back() == TAIL) 
                    {
                        float pitch, yaw;
                        memcpy(&pitch, &data[1], 4);
                        memcpy(&yaw, &data[5], 4);
                        
                        // 更新原子变量
                        latest_gimbal_pitch_ = pitch;
                        latest_gimbal_yaw_ = yaw;
                        
                        // 发布云台角度消息
                        auto msg = armor_interfaces::msg::Serial();
                        msg.header.stamp = this->now();
                        msg.yaw = yaw;
                        msg.pitch = pitch;
                        pub_->publish(msg);
                        
                        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500,
                                             "收到云台角度: yaw=%.2f°, pitch=%.2f°", yaw, pitch);
                    } 
                    else 
                    {
                        // 非预期数据，打印十六进制用于调试
                        std::string hex_str;
                        for (uint8_t byte : data) 
                        {
                            char temp[8];
                            snprintf(temp, sizeof(temp), "%02X ", byte);
                            hex_str += temp;
                        }
                        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                             "收到未知格式数据：%s", hex_str.c_str());
                    }
                }
            } 
            catch (const std::exception& e) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                     "接收异常: %s", e.what());
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SerialNode>());
    rclcpp::shutdown();
    return 0;
}