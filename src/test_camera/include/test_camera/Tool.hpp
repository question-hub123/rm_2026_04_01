#pragma once
#include <cstddef>
#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/logging.hpp>
#include <eigen3/Eigen/Dense>
#include <vector>

class Tool
{
public:
    std::vector<std::vector<cv::Point>> findContours_blue(cv::Mat& img, cv::Mat& mask2)//找轮廓
	{
        cv::Mat hsv;
        cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
        cv::Mat blue_mask;
        cv::inRange(hsv, cv::Scalar(80, 0, 60), cv::Scalar(100, 255, 255), blue_mask);

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
            //if(ratio1 < 0.5 || ratio1 > 25.0) continue;

            for (size_t j = i + 1; j < contours.size(); ++j) {
                if (used[j]) continue;

                cv::RotatedRect& r2 = rrects[j];
                float h2 = std::max(r2.size.width, r2.size.height);
                float w2 = std::min(r2.size.width, r2.size.height);
                float ratio2 = h2 / w2;
                //if (ratio2 < 3.4 || ratio2 > 25.0) continue;

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

    void drawOtherArmors(cv::Mat& img, const cv::Mat& rvec, const cv::Mat& tvec)
    {
        // ==================== 参数配置 ====================
        const double ARMOR_W = 0.135;   // 真实装甲板宽 135mm
        const double ARMOR_H = 0.055;   // 真实装甲板高 125mm
        const double CAR_RADIUS = 0.20; // 车辆半径 200mm

        const cv::Mat K = (cv::Mat_<double>(3,3) <<
        1296.16167, 0.0,        643.60901,
        0.0,        1296.23028, 509.49319,
        0.0,        0.0,        1.0);

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

    void initAnglePlot(size_t max_history = 200,
                       cv::Vec2f yaw_range = cv::Vec2f(-30.f, 30.f),
                       cv::Vec2f pitch_range = cv::Vec2f(-10.f, 10.f))
    {
        max_history_ = max_history;
        yaw_range_ = yaw_range;
        pitch_range_ = pitch_range;
        plot_yaw_   = cv::Mat::zeros(480, 640, CV_8UC3);
        plot_pitch_ = cv::Mat::zeros(480, 640, CV_8UC3);
    }

    /**
     * @brief 每帧更新并显示曲线（在主循环中调用，注意不要与其它 imshow/waitKey 冲突）
     * @param raw_yaw        原始绝对 yaw（度）
     * @param filtered_yaw   卡尔曼滤波后的 yaw（度）
     * @param raw_pitch      原始绝对 pitch（度）
     * @param filtered_pitch 卡尔曼滤波后的 pitch（度）
     */
    void updateAndPlotAngles(double raw_yaw, double filtered_yaw,
                             double raw_pitch, double filtered_pitch)
    {
        // 更新缓冲区（保持最大长度）
        raw_yaw_history_.push_back(raw_yaw);
        filtered_yaw_history_.push_back(filtered_yaw);
        raw_pitch_history_.push_back(raw_pitch);
        filtered_pitch_history_.push_back(filtered_pitch);

        if (raw_yaw_history_.size() > max_history_)       raw_yaw_history_.pop_front();
        if (filtered_yaw_history_.size() > max_history_) filtered_yaw_history_.pop_front();
        if (raw_pitch_history_.size() > max_history_)     raw_pitch_history_.pop_front();
        if (filtered_pitch_history_.size() > max_history_) filtered_pitch_history_.pop_front();

        // 绘制 yaw 曲线
        drawSingleCurve(plot_yaw_, raw_yaw_history_, filtered_yaw_history_,
                        "Yaw (deg)", yaw_range_);
        // 绘制 pitch 曲线
        drawSingleCurve(plot_pitch_, raw_pitch_history_, filtered_pitch_history_,
                        "Pitch (deg)", pitch_range_);

        cv::imshow("Yaw Curve", plot_yaw_);
        cv::imshow("Pitch Curve", plot_pitch_);
    }

    Eigen::Vector3d cameraToWorld(const Eigen::Vector3d& pos_cam, double curr_yaw, double curr_pitch) 
    {
        // 1. 将 OpenCV 相机坐标系 (x-右, y-下, z-前) 
        //    映射到云台初始坐标系 (x-前, y-左, z-上)
        //    映射关系：x_w = z_c, y_w = -x_c, z_w = -y_c
        Eigen::Vector3d p_gimbal_initial;
        p_gimbal_initial << pos_cam.z(), -pos_cam.x(), -pos_cam.y();

        // 2. 构建旋转矩阵
        // 注意：RoboMaster 的 Pitch 通常向上为正/负需根据你电控协议确定
        // 这里假设：Pitch 向上抬头为正，Yaw 向左转为正
        
        // 绕 Y 轴旋转 (Pitch)
        Eigen::AngleAxisd pitch_rot(curr_pitch, Eigen::Vector3d::UnitY());
        // 绕 Z 轴旋转 (Yaw)
        Eigen::AngleAxisd yaw_rot(curr_yaw, Eigen::Vector3d::UnitZ());

        // 组合旋转矩阵 (先绕 Pitch 旋，再绕 Yaw 旋)
        Eigen::Matrix3d rotation_matrix = yaw_rot.toRotationMatrix() * pitch_rot.toRotationMatrix();

        // 3. 执行旋转变换
        Eigen::Vector3d p_world = rotation_matrix * p_gimbal_initial;

        // 4. (可选) 补偿相机相对于云台中心的偏移量 (Offset)
        // 如果你的相机安装在云台轴心上方 5cm，前方 10cm：
        // Eigen::Vector3d offset(0.10, 0.0, 0.05);
        // p_world += offset;

        return p_world;
    }

private:
    // 绘制一张曲线图（内部辅助）
    void drawSingleCurve(cv::Mat& canvas,
                         const std::deque<double>& raw_hist,
                         const std::deque<double>& filt_hist,
                         const std::string& title,
                         cv::Vec2f range)
    {
        canvas.setTo(cv::Scalar(0, 0, 0));   // 黑色背景

        int w = canvas.cols;
        int h = canvas.rows;
        float min_val = range[0];
        float max_val = range[1];
        float scale_x = static_cast<float>(w) / (max_history_ > 1 ? max_history_ : 1);
        float scale_y = static_cast<float>(h) / (max_val - min_val);

        // 中线
        cv::line(canvas, cv::Point(0, h/2), cv::Point(w, h/2), cv::Scalar(50,50,50), 1);
        // 标题
        cv::putText(canvas, title, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2);

        // 画原始角度曲线（蓝色）
        drawLineSeries(canvas, raw_hist, scale_x, scale_y, h, min_val, cv::Scalar(255,0,0));
        // 画滤波角度曲线（绿色）
        drawLineSeries(canvas, filt_hist, scale_x, scale_y, h, min_val, cv::Scalar(0,255,0));

        // 图例
        cv::putText(canvas, "Raw", cv::Point(w-120,30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,0,0), 2);
        cv::putText(canvas, "Filtered", cv::Point(w-120,60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);
    }

    void drawLineSeries(cv::Mat& img, const std::deque<double>& data,
                        float scale_x, float scale_y, int img_h, float min_val,
                        cv::Scalar color)
    {
        if (data.size() < 2) return;
        for (size_t i = 1; i < data.size(); ++i)
        {
            int x1 = static_cast<int>((i-1) * scale_x);
            int y1 = static_cast<int>(img_h - (data[i-1] - min_val) * scale_y);
            int x2 = static_cast<int>(i * scale_x);
            int y2 = static_cast<int>(img_h - (data[i] - min_val) * scale_y);

            // 边界裁剪
            x1 = std::max(0, std::min(x1, img.cols-1));
            y1 = std::max(0, std::min(y1, img.rows-1));
            x2 = std::max(0, std::min(x2, img.cols-1));
            y2 = std::max(0, std::min(y2, img.rows-1));

            cv::line(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);
        }
    }

    // 历史数据队列（环形）
    std::deque<double> raw_yaw_history_;
    std::deque<double> filtered_yaw_history_;
    std::deque<double> raw_pitch_history_;
    std::deque<double> filtered_pitch_history_;

    // 绘图参数
    size_t max_history_ = 200;                     // 曲线最多显示点数
    cv::Vec2f yaw_range_   = cv::Vec2f(-30.f, 30.f);   // yaw 显示范围（度）
    cv::Vec2f pitch_range_ = cv::Vec2f(-10.f, 10.f);   // pitch 显示范围（度）

    // 绘图画布
    cv::Mat plot_yaw_;
    cv::Mat plot_pitch_;
};