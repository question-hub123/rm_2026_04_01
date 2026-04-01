#include "Camera.hpp"
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <memory>

using namespace std::chrono_literals;

class CameraStart : public rclcpp::Node
{
public:
    CameraStart() : Node("camera_node")
    {
        if(!camera_.init())
        {
            RCLCPP_INFO(this->get_logger(),"相机启动失败");
            rclcpp::shutdown();
            return;
        }
        RCLCPP_INFO(this->get_logger(),"相机启动成功");

        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("image_raw", 10);

        timer_ = this->create_wall_timer(33ms, std::bind(&CameraStart::timer_callback, this));
    }

private:
    // 可添加成员变量（发布者/订阅者/定时器等）
    CameraWrapper camera_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;

    void timer_callback()
    {
        cv::Mat frame;
        if(!camera_.getFrame(frame))
        {
            RCLCPP_INFO(this->get_logger(),"wrong");
            return;
        }

        if(!camera_.setExposure(10.0f))
        {
            RCLCPP_INFO(this->get_logger(), "曝光时间设置失败"); 
        }
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(),"bgr8",frame).toImageMsg();

        msg->header.stamp = this->now();
        publisher_->publish(*msg);
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraStart>());
    rclcpp::shutdown();
    return 0;
}