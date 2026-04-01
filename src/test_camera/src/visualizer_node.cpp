#include <exception>
#include <opencv2/highgui.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>

class Visualizer : public rclcpp::Node
{
public:
    Visualizer() : Node("visualizer_node")
    {
        RCLCPP_INFO(this->get_logger(), "可视化节点");
        
        sub_ = this->create_subscription<sensor_msgs::msg::Image>("image_raw", 10, std::bind(&Visualizer::image_callback, this, std::placeholders::_1));

        cv::namedWindow("aa",cv::WINDOW_NORMAL);
    }

    ~Visualizer()
    {
        cv::destroyAllWindows();
    }

private:
    // 可添加成员变量（发布者/订阅者/定时器等）
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        try 
        {
            cv::Mat frame = cv_bridge::toCvCopy(msg,"bgr8")->image;

            cv::imshow("aa",frame);
            
            int key = cv::waitKey(1);
            if(key == 27)
            {
                RCLCPP_INFO(this->get_logger(),"退出节点");
                rclcpp::shutdown();
            }
        } 
        catch (const cv_bridge::Exception& e) 
        {
            RCLCPP_INFO(this->get_logger(),"cv_bridge异常:%s",e.what());
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "标准异常: %s", e.what());
        }
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Visualizer>());
    rclcpp::shutdown();
    return 0;
}


/*#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>*/