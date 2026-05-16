#pragma once

#include <opencv2/core.hpp>

class Monitor
{
private:
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

public:
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

    ~Monitor() {
        cv::destroyAllWindows();
    }
};