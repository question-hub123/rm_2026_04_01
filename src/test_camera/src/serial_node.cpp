#include "rclcpp/rclcpp.hpp"
#include "armor_interfaces/msg/armor_array.hpp"
#include "serial_driver/serial_driver.hpp"
#include "cstring"
#include "opencv2/opencv.hpp"

#define TX_FRAME_LEN 10
#define HEAD 0x00
#define TAIL 0x01

class SerialNode : public rclcpp::Node
{
public:
    SerialNode() : Node("serial_node")
    {
        RCLCPP_INFO(this->get_logger(), "Node has been started.");

        const std::string port = "/dev/ttyUSB0";
        const int baud = 115200;

        try {
            owned_ctx_ = std::make_unique<IoContext>(2);
            serial_driver_ = std::make_unique<drivers::serial_driver::SerialDriver>(*owned_ctx_);
            
            drivers::serial_driver::SerialPortConfig config(
                static_cast<uint32_t>(baud),
                drivers::serial_driver::FlowControl::NONE,
                drivers::serial_driver::Parity::NONE,
                drivers::serial_driver::StopBits::ONE
            );
            
            serial_driver_->init_port(port, config);
            if (!serial_driver_->port()->is_open()) {
                serial_driver_->port()->open();
            }
            
            RCLCPP_INFO(this->get_logger(), "串口初始化成功！");
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "串口打开失败：%s", e.what());
            return;
        }

        sub_ = this->create_subscription<armor_interfaces::msg::ArmorArray>(
            "armor_msgs_filtered", 10,
            std::bind(&SerialNode::callback, this, std::placeholders::_1)
        );
    }

    ~SerialNode()
    {
        if (serial_driver_ && serial_driver_->port()->is_open()) {
            serial_driver_->port()->close();
        }
        if (owned_ctx_) {
            owned_ctx_->waitForExit();
        }
    }

private:
    rclcpp::Subscription<armor_interfaces::msg::ArmorArray>::SharedPtr sub_;
    std::unique_ptr<IoContext> owned_ctx_;
    std::unique_ptr<drivers::serial_driver::SerialDriver> serial_driver_;

    void callback(armor_interfaces::msg::ArmorArray::SharedPtr msg)
    {
        if (!serial_driver_ || !serial_driver_->port()->is_open()) return;

        if(msg->armors.empty())
        {
            RCLCPP_INFO(this->get_logger(), "未识别到装甲板");
            return;
        }

        const auto& armor = msg->armors[0];

        float yaw_rad = armor.yaw_filtered;
        float pitch_rad = armor.pitch;
        float yaw_deg = yaw_rad * 180.0 / CV_PI;
        float pitch_deg = pitch_rad * 180.0 / CV_PI;

        uint8_t buf[TX_FRAME_LEN] = {0};
        buf[0] = HEAD;
        memcpy(&buf[1], &pitch_deg, 4);
        memcpy(&buf[5], &yaw_deg, 4);
        buf[9] = TAIL;

        std::vector<uint8_t> send_data(buf, buf + TX_FRAME_LEN);
        serial_driver_->port()->send(send_data);

        RCLCPP_INFO(this->get_logger(), "yaw_deg=%.2f, pitch_deg=%.2f", yaw_deg, pitch_deg);
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SerialNode>());
    rclcpp::shutdown();
    return 0;
}
