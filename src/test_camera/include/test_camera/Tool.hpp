#pragma once

#include <cstddef>
#include <cstdlib>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/logging.hpp>
#include <vector>
#include <Eigen/Dense>

class Tool
{
public:
    std::vector<std::vector<cv::Point>> findContours2_blue(cv::Mat& img, cv::Mat& mask2)//找轮廓
	{
        cv::Mat hsv;
        cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
        cv::Mat blue_mask;
        cv::inRange(hsv, cv::Scalar(80, 50, 50), cv::Scalar(100, 255, 255), blue_mask);

        /*cv::Mat white_mask;
        cv::inRange(hsv, cv::Scalar(0, 0, 200), cv::Scalar(180, 30, 255), white_mask);
        cv::bitwise_and(blue_mask, ~white_mask, blue_mask);*/

		cv::Mat gray, mask;
		cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
		cv::GaussianBlur(gray, gray, cv::Size(3, 3), 1.2);
		cv::threshold(gray, mask, 150, 255, cv::THRESH_BINARY);
        
        cv::Mat final_mask; 
        cv::bitwise_and(blue_mask, mask, final_mask);
        mask2 = final_mask;


		std::vector<std::vector<cv::Point>>contours;
		std::vector<std::vector<cv::Point>>final_contours;
		cv::findContours(final_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		for (size_t i = 0; i < contours.size(); ++i)
		{
			double area = cv::contourArea(contours[i]);
			if (area < 100) continue;
			cv::Rect rect = cv::boundingRect(contours[i]);
			cv::Point2f rect_center(rect.x + rect.width / 2, rect.y + rect.height / 2);
			cv::drawContours(img, contours, i, cv::Scalar(0, 255, 0), 2);
			final_contours.push_back(contours[i]);
		}
		return final_contours;
	}

    std::vector<std::vector<cv::Point>> findContours2_red(cv::Mat& img, cv::Mat& mask2)//找轮廓
	{
        cv::Mat hsv;
        cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
        cv::Mat red_mask;
        cv::Mat mask_low, mask_high;

        cv::inRange(hsv, cv::Scalar(0, 0, 0), cv::Scalar(125, 255, 255), mask_low);
        cv::inRange(hsv, cv::Scalar(100, 0, 0), cv::Scalar(180, 255, 255), mask_high);

        /*cv::inRange(hsv, cv::Scalar(0, 50, 0), cv::Scalar(45, 255, 255), mask_low);
        cv::inRange(hsv, cv::Scalar(100, 70, 100), cv::Scalar(180, 255, 255), mask_high);*/
        cv::bitwise_or(mask_low, mask_high, red_mask);

        cv::Mat white_mask;
        cv::inRange(hsv, cv::Scalar(0, 0, 150), cv::Scalar(180, 30, 255), white_mask);
        cv::bitwise_and(red_mask, ~white_mask, red_mask);

		cv::Mat gray, mask;
		cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
		cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.2);
		cv::threshold(gray, mask, 150, 255, cv::THRESH_BINARY);
        
        cv::Mat final_mask; 
        cv::bitwise_and(red_mask, mask, final_mask);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::dilate(final_mask, final_mask, kernel);

        mask2 = final_mask;

		std::vector<std::vector<cv::Point>>contours;
		std::vector<std::vector<cv::Point>>final_contours;
		cv::findContours(final_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		for (size_t i = 0; i < contours.size(); ++i)
		{
			double area = cv::contourArea(contours[i]);
			if (area < 120) continue;
			cv::Rect rect = cv::boundingRect(contours[i]);
			cv::Point2f rect_center(rect.x + rect.width / 2, rect.y + rect.height / 2);
			cv::drawContours(img, contours, i, cv::Scalar(0, 255, 0), 2);
			final_contours.push_back(contours[i]);
		}
		return final_contours;
	}

    std::vector<std::vector<cv::Point2f>> drawRect3(std::vector<std::vector<cv::Point>> contours, cv::Mat& img)
    {
        std::vector<bool> used(contours.size(), false);
        std::vector<std::pair<int, int>> armorGroups;
        std::vector<std::vector<cv::Point2f>> armorCorners;

        // 预先计算每个轮廓的旋转矩形（避免重复计算）
        std::vector<cv::RotatedRect> rrects(contours.size());
        for (size_t i = 0; i < contours.size(); ++i) {
            rrects[i] = cv::minAreaRect(contours[i]);
        }

        // 配对灯条
        for (size_t i = 0; i < contours.size(); ++i) {
            if (used[i]) continue;

            cv::RotatedRect& r1 = rrects[i];
            float h1 = std::max(r1.size.width, r1.size.height);   // 灯条高度（长边）
            float w1 = std::min(r1.size.width, r1.size.height);   // 灯条宽度（短边）
            float ratio1 = h1 / w1;
            // 长宽比过滤，排除明显不是灯条的轮廓
            //if (ratio1 < 0.5 || ratio1 > 25.0) continue;
            if(ratio1 < 1.0 || ratio1 > 25.0) continue;

            for (size_t j = i + 1; j < contours.size(); ++j) {
                if (used[j]) continue;

                cv::RotatedRect& r2 = rrects[j];
                float h2 = std::max(r2.size.width, r2.size.height);
                float w2 = std::min(r2.size.width, r2.size.height);
                float ratio2 = h2 / w2;
                if (ratio2 < 1.0 || ratio2 > 25.0) continue;

                float avg_h = (h1 + h2) / 2.0f;

                // 高度差不能太大（通常不超过平均高度的1/4）
                if (std::abs(h1 - h2) > avg_h * 1.75f) continue;

                // Y方向中心点偏差不能太大
                if (std::abs(r1.center.y - r2.center.y) > avg_h * 0.75f) continue;

                // 水平距离应在合理范围内（1.2倍到5倍平均高度）
                float x_diff = std::abs(r1.center.x - r2.center.x);
                if (x_diff < avg_h * 1.5f || x_diff > avg_h * 10.0f) continue;

                //倾斜度
                double angle1 = r1.angle;
                double angle2 = r2.angle;
                //if(std::abs(angle1 - angle2) > 50.5f) continue;

                // 通过所有条件，配对成功
                armorGroups.emplace_back(i, j);
                used[i] = true;
                used[j] = true;
                break; // 找到第一个符合条件的就配对，如需更优配对可改为评分选最佳
            }
        }

        // 绘制装甲板（与 drawRect 中的绘制逻辑类似）
        for (auto& group : armorGroups) {
            int idx1 = group.first;
            int idx2 = group.second;

            cv::RotatedRect& leftRect = (rrects[idx1].center.x < rrects[idx2].center.x) ? rrects[idx1] : rrects[idx2];
            cv::RotatedRect& rightRect = (rrects[idx1].center.x < rrects[idx2].center.x) ? rrects[idx2] : rrects[idx1];

            // 获取左右灯条的上下端点（使用旋转矩形的四个顶点）
            cv::Point2f topLeft, bottomLeft, topRight, bottomRight;

            cv::Point2f leftPts[4];
            leftRect.points(leftPts);
            std::sort(leftPts, leftPts + 4, [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
            topLeft = (leftPts[0] + leftPts[1]) / 2.0f;
            bottomLeft = (leftPts[2] + leftPts[3]) / 2.0f;

            cv::Point2f rightPts[4];
            rightRect.points(rightPts);
            std::sort(rightPts, rightPts + 4, [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
            topRight = (rightPts[0] + rightPts[1]) / 2.0f;
            bottomRight = (rightPts[2] + rightPts[3]) / 2.0f;

            // 可选：向心收缩（与 drawRect 一致）
            const float SHRINK_RATIO = 0.80f;
            auto shrink = [SHRINK_RATIO](cv::Point2f& top, cv::Point2f& bottom) {
                cv::Point2f center = (top + bottom) / 2.0f;
                top = center + (top - center) * SHRINK_RATIO;
                bottom = center + (bottom - center) * SHRINK_RATIO;
            };
            shrink(topLeft, bottomLeft);
            shrink(topRight, bottomRight);

            std::vector<cv::Point2f> corners = { topLeft, topRight, bottomRight, bottomLeft };
            armorCorners.push_back(corners);

            // 绘制角点和边框
            for (auto& corner : corners) {
                cv::circle(img, corner, 4, cv::Scalar(0, 0, 255), -1);
            }
            cv::line(img, corners[0], corners[1], cv::Scalar(0, 255, 0), 2);
            cv::line(img, corners[1], corners[2], cv::Scalar(0, 255, 0), 2);
            cv::line(img, corners[2], corners[3], cv::Scalar(0, 255, 0), 2);
            cv::line(img, corners[3], corners[0], cv::Scalar(0, 255, 0), 2);
        }

        return armorCorners;
    }



    void drawOtherArmors(cv::Mat& img, const cv::Mat& rvec, const cv::Mat& tvec)
    {
        // ==================== 1. 参数配置 ====================
        const double ARMOR_W = 0.135; 
        const double ARMOR_H = 0.125; 
        const double CAR_RADIUS = 0.20; 
        const cv::Mat K = (cv::Mat_<double>(3,3) << 2374.54248, 0.0, 698.85288, 0.0, 2377.53648, 520.8649, 0.0, 0.0, 1.0);

        // ==================== 2. 提取状态并平滑 Pitch ====================
        cv::Mat R_cur;
        cv::Rodrigues(rvec, R_cur);
        
        // 获取 Yaw
        cv::Mat normal = R_cur * (cv::Mat_<double>(3,1) << 0, 0, 1);
        double yaw = std::atan2(normal.at<double>(0), normal.at<double>(2));

        // 获取原始的向上向量（包含 Pitch 和 Roll）
        cv::Mat armor_up_vec = R_cur * (cv::Mat_<double>(3,1) << 0, 1, 0); 
        double hx = armor_up_vec.at<double>(0);
        double hy = armor_up_vec.at<double>(1);
        double hz = armor_up_vec.at<double>(2);

        // --- 【新增：消除 0 Pitch 抖动逻辑】 ---
        double cos_y = std::cos(yaw);
        double sin_y = std::sin(yaw);

        // 步骤 A: 将向量反向旋转抵消 Yaw，转入“车体局部坐标系”
        // 在这个坐标系下，hz_local 纯粹代表前后俯仰，hx_local 纯粹代表左右侧倾
        double hx_local = cos_y * hx - sin_y * hz;
        double hz_local = sin_y * hx + cos_y * hz;
        double hy_local = hy;

        // 步骤 B: 钳制 Pitch（hz_local < 0 代表装甲板顶部远离相机，即上翻）
        // 设定 0.05 的阈值（大约 3 度），如果在这个范围内抖动，强制锁定为上翻
        const double PITCH_THRESH = -0.20; 
        if (std::abs(hz_local) < 2.0) {
            hz_local = PITCH_THRESH; 
        }

        // （可选隐藏福利）: 如果你发现重投影有左右倾斜抖动，可以解除下面这行的注释，强制消除侧倾
        hx_local = 0.0; 

        // 步骤 C: 重新归一化（防止强制修改后向量长度变化导致框变大/变小）
        double len = std::sqrt(hx_local*hx_local + hy_local*hy_local + hz_local*hz_local);
        hx_local /= len; 
        hy_local /= len; 
        hz_local /= len;

        // 步骤 D: 带着干净的 Pitch 重新旋转回相机坐标系，作为全局高度向量
        cv::Point3d global_h_dir(
            cos_y * hx_local + sin_y * hz_local,
            hy_local,
            -sin_y * hx_local + cos_y * hz_local
        );
        // ------------------------------------

        // ==================== 3. 计算车体中心 ====================
        double car_center_cam_x = tvec.at<double>(0) - CAR_RADIUS * std::sin(yaw);
        double car_center_cam_y = tvec.at<double>(1); 
        double car_center_cam_z = tvec.at<double>(2) - CAR_RADIUS * std::cos(yaw);

        // ==================== 4. 绘制剩下三块板 ====================
        for (int i = 1; i <= 3; ++i) { 
            double current_yaw = yaw + i * (CV_PI / 2.0);

            double cx = car_center_cam_x + CAR_RADIUS * std::sin(current_yaw);
            double cy = car_center_cam_y;
            double cz = car_center_cam_z + CAR_RADIUS * std::cos(current_yaw);

            double cur_cos_y = std::cos(current_yaw);
            double cur_sin_y = std::sin(current_yaw);
            cv::Point3d w_dir(cur_cos_y, 0, -cur_sin_y); 

            // 使用过滤后的 global_h_dir 代替原来的 h_dir
            std::vector<cv::Point3d> corners_cam;
            corners_cam.push_back(cv::Point3d(cx, cy, cz) - w_dir*(ARMOR_W/2.0) - global_h_dir*(ARMOR_H/2.0));
            corners_cam.push_back(cv::Point3d(cx, cy, cz) + w_dir*(ARMOR_W/2.0) - global_h_dir*(ARMOR_H/2.0));
            corners_cam.push_back(cv::Point3d(cx, cy, cz) + w_dir*(ARMOR_W/2.0) + global_h_dir*(ARMOR_H/2.0));
            corners_cam.push_back(cv::Point3d(cx, cy, cz) - w_dir*(ARMOR_W/2.0) + global_h_dir*(ARMOR_H/2.0));

            std::vector<cv::Point2f> img_pts;
            for (const auto& p : corners_cam) {
                if (p.z <= 0) continue;
                float u = K.at<double>(0,0) * p.x / p.z + K.at<double>(0,2);
                float v = K.at<double>(1,1) * p.y / p.z + K.at<double>(1,2);
                img_pts.push_back(cv::Point2f(u, v));
            }

            if (img_pts.size() == 4) {
                for (int j = 0; j < 4; ++j)
                    cv::line(img, img_pts[j], img_pts[(j+1)%4], cv::Scalar(255, 100, 0), 2);
            }
        }
    }

        void drawOtherArmors_ori(cv::Mat& img, const cv::Mat& rvec, const cv::Mat& tvec)
    {
        // ==================== 参数配置 ====================
        const double ARMOR_W = 0.135;   // 真实装甲板宽 135mm
        const double ARMOR_H = 0.085;   // 真实装甲板高 125mm
        const double CAR_RADIUS = 0.20; // 车辆半径 200mm

        const cv::Mat K = (cv::Mat_<double>(3,3) << 1330.54525, 0.0, 642.60771,
                0.0,        1329.21216, 492.68961,
                0.0,        0.0,        1.0);;

        // 1. 提取原始 Yaw
        cv::Mat R_cur;
        cv::Rodrigues(rvec, R_cur);
        cv::Mat normal = R_cur * (cv::Mat_<double>(3,1) << 0, 0, 1);
        double raw_yaw = std::atan2(normal.at<double>(0), normal.at<double>(2));

        // --- 【新增：PnP 抽搐抑制器】 ---
        // PnP在正对时极易出现左右各十几度的跳动，这里加一个软钳制
        // 如果装甲板差不多是正对的，强制让 Yaw 归零或减弱，防止车体中心疯狂乱甩
        double yaw = raw_yaw;
        if (std::abs(yaw) < 0.25) { // 大约 15 度以内
            yaw *= 0.5; // 削弱 50% 的抖动幅度
        }

        // 2. 稳定提取平移向量
        double tx = tvec.at<double>(0);
        double ty = tvec.at<double>(1);
        double tz = tvec.at<double>(2);

        // 计算车体中心 (相机坐标系)
        // 注意：这里基于平滑后的 yaw 计算中心，中心点会稳定很多
        double cx = tx - CAR_RADIUS * std::sin(yaw);
        double cy = ty; 
        double cz = tz - CAR_RADIUS * std::cos(yaw);

        // 3. 循环绘制 4 块板
        for (int i = 0; i < 4; ++i) {
            double current_yaw = yaw + i * (CV_PI / 2.0);

            // A. 该板的中心点
            double board_cx = cx + CAR_RADIUS * std::sin(current_yaw);
            double board_cy = cy;
            double board_cz = cz + CAR_RADIUS * std::cos(current_yaw);

            // B. 计算该板的四个角点 (严格按照 OpenCV 坐标系：X右，Y下，Z前)
            std::vector<cv::Point3d> corners_cam;
            double cos_y = std::cos(current_yaw);
            double sin_y = std::sin(current_yaw);

            // 左右延伸向量 (X-Z平面)
            cv::Point3d w_vec(cos_y * (ARMOR_W / 2.0), 0.0, -sin_y * (ARMOR_W / 2.0));
            // 上下延伸向量 (纯粹的 OpenCV Y轴，Y向下为正，所以 -h_vec 是往上)
            cv::Point3d h_vec(0.0, ARMOR_H / 2.0, 0.0);

            corners_cam.push_back(cv::Point3d(board_cx, board_cy, board_cz) - w_vec - h_vec); // 左上
            corners_cam.push_back(cv::Point3d(board_cx, board_cy, board_cz) + w_vec - h_vec); // 右上
            corners_cam.push_back(cv::Point3d(board_cx, board_cy, board_cz) + w_vec + h_vec); // 右下
            corners_cam.push_back(cv::Point3d(board_cx, board_cy, board_cz) - w_vec + h_vec); // 左下

            // C. 投影并绘制
            std::vector<cv::Point2f> img_pts;
            bool out_of_bounds = false;
            for (const auto& p : corners_cam) {
                if (p.z <= 0.1) { // 防止 Z <= 0 导致除以 0 画面崩溃
                    out_of_bounds = true;
                    break;
                }
                float u = K.at<double>(0,0) * p.x / p.z + K.at<double>(0,2);
                float v = K.at<double>(1,1) * p.y / p.z + K.at<double>(1,2);
                img_pts.push_back(cv::Point2f(u, v));
            }

            if (!out_of_bounds && img_pts.size() == 4) {
                // 当前装甲板画绿色，其他画蓝色
                cv::Scalar color = (i == 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 150, 0);
                int thickness = (i == 0) ? 3 : 2;
                
                for (int j = 0; j < 4; ++j) {
                    cv::line(img, img_pts[j], img_pts[(j+1)%4], color, thickness, cv::LINE_AA);
                }
            }
        }
    }
    
    
};