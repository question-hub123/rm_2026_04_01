#include <cmath>
#include <cstddef>
#include <deque>
#include <memory>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/time.hpp>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/dnn.hpp>

#include "armor_interfaces/msg/armor.hpp"
#include "armor_interfaces/msg/armor_array.hpp"
#include "Tool.hpp"
#include "Pnp.hpp"
#include "Camera.hpp"

using namespace std::chrono_literals;
const float CONF_THRESH = 0.0f;
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
        camera_->setExposure(5.0);

        pub_ = this->create_publisher<armor_interfaces::msg::ArmorArray>("armor_msgs", 10);
        timer_ = this->create_wall_timer(33ms, std::bind(&ArmorDetctor::timer_callback,this));


        std::string model_path = "/home/aa/rm_ws/Zenet-已训练好.onnx";
        try 
        {
            net_ = cv::dnn::readNetFromONNX(model_path);
            classifier_loaded_ = true;
            RCLCPP_INFO(this->get_logger(), "ONNX模型加载成功");
        } 
        catch (const cv::Exception& e) 
        {
            RCLCPP_ERROR(this->get_logger(), "模型加载失败: %s", e.what());
        }
    }

private:
    // 可添加成员变量（发布者/订阅者/定时器等）
    rclcpp::Publisher<armor_interfaces::msg::ArmorArray>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::unique_ptr<Tool> tool_;
    std::unique_ptr<PnpSolver> pnpsolver_;
    std::unique_ptr<CameraWrapper> camera_;

    //数字识别功能
    // 类成员
    cv::dnn::Net net_;                     // 与原代码 net_ 对应
    const cv::Size MODEL_INPUT_SIZE = cv::Size(28, 28);
    bool classifier_loaded_ = false;


    int recognizeDigit(const cv::Mat& src, const std::vector<cv::Point2f>& corners)
    {
        if (net_.empty() || corners.size() != 4) return -1;

        const float vertical_padding_ratio = 0.6f;

        // ===================== 1. 调整角点顺序以匹配原代码逻辑 =====================
        // 原代码假设 pnp_corners 顺序为：0左上, 1左下, 2右下, 3右上
        // 你的 corners 通常为：0左上, 1右上, 2右下, 3左下
        // 因此需要重新排列
        std::vector<cv::Point2f> roi_corners(4);
        roi_corners[0] = corners[0];                 // 左上
        roi_corners[1] = corners[3];                 // 左下（原代码的1）
        roi_corners[2] = corners[2];                 // 右下（原代码的2）
        roi_corners[3] = corners[1];                 // 右上（原代码的3）

        // ===================== 2. 垂直方向扩展 ROI =====================
        float left_height  = cv::norm(roi_corners[0] - roi_corners[1]);
        float right_height = cv::norm(roi_corners[3] - roi_corners[2]);
        float avg_height = (left_height + right_height) / 2.0f;
        float y_offset = avg_height * vertical_padding_ratio;

        roi_corners[0].y -= y_offset;
        roi_corners[3].y -= y_offset;
        roi_corners[1].y += y_offset;
        roi_corners[2].y += y_offset;

        // ===================== 3. 透视变换至 28×28 =====================
        std::vector<cv::Point2f> dst_pts = {
            {0.0f, 0.0f},
            {0.0f, (float)MODEL_INPUT_SIZE.height},
            {(float)MODEL_INPUT_SIZE.width, (float)MODEL_INPUT_SIZE.height},
            {(float)MODEL_INPUT_SIZE.width, 0.0f}
        };

        cv::Mat warp_mat = cv::getPerspectiveTransform(roi_corners, dst_pts);
        cv::Mat roi_for_model;
        cv::warpPerspective(src, roi_for_model, warp_mat, MODEL_INPUT_SIZE);

        // ===================== 4. 灰度化 =====================
        cv::Mat gray;
        cv::cvtColor(roi_for_model, gray, cv::COLOR_BGR2GRAY);

        // ===================== 5. 构建 blob 并推理 =====================
        cv::Mat blob = cv::dnn::blobFromImage(gray, 1.0 / 255.0, MODEL_INPUT_SIZE, cv::Scalar(0), false, false);
        net_.setInput(blob);
        cv::Mat prob = net_.forward();

        // ===================== 6. 后处理（完全照搬原代码） =====================
        // 提取分类概率（忽略背景类，取第1列到最后一列对应数字0-9）
        cv::Mat cls_prob = prob.colRange(1, prob.cols);
        
        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(cls_prob.reshape(1, 1), nullptr, &confidence, nullptr, &classIdPoint);

        // classIdPoint.x 对应数字 0-9
        return classIdPoint.x;
    }



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
        auto contours = tool_->findContours_blue(img, mask2);
        auto armorCornersList = tool_->drawRect(contours, img);

        std::vector<DetectedArmor> detections;

        for(auto& corners : armorCornersList)
        {
            if(corners.size() != 4) continue;

            armor_interfaces::msg::Armor armor_msg;


            cv::Mat rvec, tvec;
            double yaw, pitch, distance;
            if (pnpsolver_->solveWithPose(corners, rvec, tvec, yaw, pitch, distance))
            {
                DetectedArmor da;
                da.corners = corners;
                da.x = tvec.at<double>(0) * 1000.0;   // mm
                da.y = tvec.at<double>(1) * 1000.0;
                da.z = tvec.at<double>(2) * 1000.0;

                double yaw_target = std::atan2(tvec.at<double>(0),tvec.at<double>(2)) * 180.0 / M_PI;
                double pitch_target = std::atan2(tvec.at<double>(1),tvec.at<double>(2)) * 180.0 / M_PI;

                da.yaw = yaw_target;
                da.pitch = pitch_target;
                da.rvec = rvec;
                da.tvec = tvec;
                da.valid = true;
                detections.push_back(da);
                
                cv::Point2f center = (corners[0] + corners[2]) / 2.0f;

                int digit = recognizeDigit(img, corners);
                if(digit != -1)
                {
                    cv::putText(img, "Type : " + std::to_string(digit), center + cv::Point2f(-150, -150) , cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
                }


                int test_y = -100;
                cv::Point text_pos(center.x - 80, center.y + test_y);
                tool_->drawOtherArmors(img, rvec, tvec);
                //tool_->drawCarCenter(img, rvec, tvec);
                std::cout<<"Yaw : "<<rvec.at<double>(0)<<" Pitch: "<<rvec.at<double>(1)<<std::endl;
                cv::putText(img, "Yaw: " + std::to_string(yaw_target), text_pos,cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
                cv::putText(img, "Pitch: " + std::to_string(pitch_target), text_pos + cv::Point(0,75),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
                cv::putText(img, "Distance: " + std::to_string(distance), text_pos + cv::Point(0,90),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
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

            // 填充位姿信息
            armor_msg.yaw = detections[i].yaw;
            armor_msg.pitch = detections[i].pitch;
            armor_msg.yaw_filtered = detections[i].yaw;   // 原始值，滤波节点会替换
            armor_msg.pitch_filtered = detections[i].pitch;

            armor_array_msg.armors.push_back(armor_msg);

            if (!detections[i].corners.empty())
            {
                cv::Point2f center = (detections[i].corners[0] + detections[i].corners[2]) / 2.0;
                //cv::putText(img, "ID: " + std::to_string(assigned_ids[i]), center, cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 2);
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