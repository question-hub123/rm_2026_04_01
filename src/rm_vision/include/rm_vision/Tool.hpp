#pragma once
#include <cstddef>
#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/logging.hpp>
#include <vector>

class Tool
{
public:
    std::vector<std::vector<cv::Point>> findContours_blue(cv::Mat& img, cv::Mat& mask2)//找轮廓
	{
        cv::Mat hsv;
        cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
        cv::Mat blue_mask;
        cv::inRange(hsv, cv::Scalar(80, 0, 100), cv::Scalar(100, 255, 255), blue_mask);

        cv::Mat white_mask;
        cv::inRange(hsv, cv::Scalar(0, 0, 200), cv::Scalar(180, 30, 255), white_mask);
        cv::bitwise_and(blue_mask, ~white_mask, blue_mask);

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
			if (area < 10) continue;
			cv::Rect rect = cv::boundingRect(contours[i]);
			cv::Point2f rect_center(rect.x + rect.width / 2, rect.y + rect.height / 2);
			cv::drawContours(img, contours, i, cv::Scalar(0, 255, 0), 2);
			final_contours.push_back(contours[i]);
		}
		return final_contours;
	}

    std::vector<std::vector<cv::Point>> findContours_red(cv::Mat& img, cv::Mat& mask2)//找轮廓
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
        cv::inRange(hsv, cv::Scalar(0, 0, 200), cv::Scalar(180, 30, 255), white_mask);
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
			if (area < 35) continue;
			cv::Rect rect = cv::boundingRect(contours[i]);
			cv::Point2f rect_center(rect.x + rect.width / 2, rect.y + rect.height / 2);
			cv::drawContours(img, contours, i, cv::Scalar(0, 255, 0), 2);
			final_contours.push_back(contours[i]);
		}
		return final_contours;
	}

    std::vector<std::vector<cv::Point2f>> drawRect(std::vector<std::vector<cv::Point>> contours, cv::Mat& img)
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
            if(ratio1 < 3.4 || ratio1 > 25.0) continue;

            for (size_t j = i + 1; j < contours.size(); ++j) {
                if (used[j]) continue;

                cv::RotatedRect& r2 = rrects[j];
                float h2 = std::max(r2.size.width, r2.size.height);
                float w2 = std::min(r2.size.width, r2.size.height);
                float ratio2 = h2 / w2;
                if (ratio2 < 3.4 || ratio2 > 25.0) continue;

                float avg_h = (h1 + h2) / 2.0f;

                // 高度差不能太大（通常不超过平均高度的1/4）
                //if (std::abs(h1 - h2) > avg_h * 1.75f) continue;

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
};