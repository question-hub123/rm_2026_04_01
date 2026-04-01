#include <memory>
#include<rclcpp/rclcpp.hpp>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "armor_interfaces/msg/armor.hpp"
#include "armor_interfaces/msg/armor_array.hpp"
#include "Tool.hpp"
#include "Pnp.hpp"
#include "Camera.hpp"

using namespace std::chrono_literals;
class ArmorDetctor : public rclcpp::Node
{
public:
    ArmorDetctor() : Node("armor_detctor_node")
    {
        RCLCPP_INFO(this->get_logger(), "装甲板检测");

        tool_ = std::make_unique<Tool>();
        pnpsolver_ = std::make_unique<PnpSolver>();
        camera_ = std::make_unique<CameraWrapper>();
        if(!camera_->init(0))
        {
            RCLCPP_ERROR(this->get_logger(), "相机打开失败，节点退出");
            rclcpp::shutdown();
            return;
        }
        camera_->setExposure(8.0);

        pub_ = this->create_publisher<armor_interfaces::msg::ArmorArray>("armor_msgs", 10);
        timer_ = this->create_wall_timer(33ms, std::bind(&ArmorDetctor::timer_callback,this));
    }

private:
    // 可添加成员变量（发布者/订阅者/定时器等）
    rclcpp::Publisher<armor_interfaces::msg::ArmorArray>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::unique_ptr<Tool> tool_;
    std::unique_ptr<PnpSolver> pnpsolver_;
    std::unique_ptr<CameraWrapper> camera_;

    struct DetectedArmor
    {
        std::vector<cv::Point2f> corners;
        double x, y, z;      // 位置 (mm)
        double yaw, pitch;
        cv::Mat rvec, tvec;
        bool valid;
    };

    struct TrackedTarget
    {
        int id;
        double x,y,z;
        double yaw;
        rclcpp::Time last_seen;
    };

    std::vector<TrackedTarget> prev_targets_;   // 上一帧目标列表
    int next_id_ = 1;

    void timer_callback()
    {
        cv::Mat img;
        if(!camera_->getFrame(img,1000))
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "获取图像帧失败");
            return;
        }

        cv::Mat mask2;
        auto contours = tool_->findContours_red(img, mask2);
        auto armorCornersList = tool_->drawRect(contours, img);

        std::vector<DetectedArmor> detections;

        for(auto& corners : armorCornersList)
        {
            if(corners.size() != 4) continue;

            armor_interfaces::msg::Armor armor_msg;

            for(auto& pt : corners)
            {
                geometry_msgs::msg::Point p;
                p.x = pt.x; p.y = pt.y; p.z = 0.0;
                armor_msg.corners.push_back(p);
            }

            cv::Mat rvec, tvec;
            double yaw, pitch, distance;
            if (pnpsolver_->solveWithPose(corners, rvec, tvec, yaw, pitch, distance))
            {
                DetectedArmor da;
                da.corners = corners;
                da.x = tvec.at<double>(0) * 1000.0;   // mm
                da.y = tvec.at<double>(1) * 1000.0;
                da.z = tvec.at<double>(2) * 1000.0;
                da.yaw = yaw;
                da.pitch = pitch;
                da.rvec = rvec;
                da.tvec = tvec;
                da.valid = true;
                detections.push_back(da);

                /*
                cv::putText(img, "Yaw: " + std::to_string(yaw), cv::Point2f(10, 30),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
		        cv::putText(img, "Pitch: " + std::to_string(pitch), cv::Point2f(10, 80),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
		        cv::putText(img, "Distance: " + std::to_string(distance), cv::Point2f(10, 130),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
                */

                cv::Point2f center = (corners[0] + corners[2]) / 2.0f;

                int test_y = -60;
                cv::Point text_pos(center.x - 80, center.y + test_y);

                cv::putText(img, "Yaw: " + std::to_string(yaw), text_pos,cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
                cv::putText(img, "Pitch: " + std::to_string(pitch), text_pos + cv::Point(0,25),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
                cv::putText(img, "Distance: " + std::to_string(distance), text_pos + cv::Point(0,50),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            }
            else
            {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "PnP解算失败");
            }
        }
        

        if (detections.empty())
        {
            RCLCPP_INFO(this->get_logger(),"NO TARGET");
        }

        std::vector<int> assigned_ids(detections.size(), -1);
        std::vector<bool> used_prev(prev_targets_.size(), false);
        rclcpp::Time now = this->now();
        double max_age_seconds = 0.5;

        std::vector<TrackedTarget> filtered_prev;
        for (auto& t : prev_targets_)
        {
            if ((now - t.last_seen).seconds() < max_age_seconds)
            {
                filtered_prev.push_back(t);
            }
        }
        prev_targets_ = filtered_prev;

        for (size_t i = 0; i < detections.size(); ++i)
        {
            double best_dist = 300.0;
            int best_idx = -1;
            for (size_t j = 0; j < prev_targets_.size(); ++j)
            {
                if (used_prev[j]) continue;
                double dx = detections[i].x - prev_targets_[j].x;
                double dy = detections[i].y - prev_targets_[j].y;
                double dist = std::sqrt(dx*dx + dy*dy);
                if (dist < best_dist)
                {
                    best_dist = dist;
                    best_idx = j;
                }
            }
            if (best_idx != -1)
            {
                assigned_ids[i] = prev_targets_[best_idx].id;
                used_prev[best_idx] = true;
            }
            else
            {
                // 新目标，分配新 ID
                assigned_ids[i] = next_id_++;
            }
        }

        std::vector<TrackedTarget> curr_targets;
        for (size_t i = 0; i < detections.size(); ++i)
        {
            TrackedTarget tt;
            tt.id = assigned_ids[i];
            tt.x = detections[i].x;
            tt.y = detections[i].y;
            tt.z = detections[i].z;
            tt.yaw = detections[i].yaw;
            tt.last_seen = now;
            curr_targets.push_back(tt);
        }

        armor_interfaces::msg::ArmorArray armor_array_msg;
        armor_array_msg.header.stamp = now;
        armor_array_msg.header.frame_id = "camera";

        for (size_t i = 0; i < detections.size(); ++i)
        {
            armor_interfaces::msg::Armor armor_msg;
            armor_msg.id = assigned_ids[i];   // 关键：填入稳定 ID

            // 填充角点
            for (auto& pt : detections[i].corners)
            {
                geometry_msgs::msg::Point p;
                p.x = pt.x; p.y = pt.y; p.z = 0.0;
                armor_msg.corners.push_back(p);
            }

            // 填充位姿信息
            armor_msg.position.x = detections[i].x;
            armor_msg.position.y = detections[i].y;
            armor_msg.position.z = detections[i].z;
            armor_msg.rotation.x = detections[i].rvec.at<double>(0);
            armor_msg.rotation.y = detections[i].rvec.at<double>(1);
            armor_msg.rotation.z = detections[i].rvec.at<double>(2);
            armor_msg.yaw = detections[i].yaw;
            armor_msg.pitch = detections[i].pitch;
            armor_msg.yaw_filtered = detections[i].yaw;   // 原始值，滤波节点会替换

            armor_array_msg.armors.push_back(armor_msg);

            if (!detections[i].corners.empty())
            {
                cv::Point2f center = (detections[i].corners[0] + detections[i].corners[2]) / 2.0;
                cv::putText(img, "ID: " + std::to_string(assigned_ids[i]), center, cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
            }
        }
        
        prev_targets_ = curr_targets;

        cv::imshow("aa",img);
        cv::imshow("bb",mask2);
        cv::waitKey(1);
        pub_->publish(armor_array_msg);
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArmorDetctor>());
    rclcpp::shutdown();
    return 0;
}