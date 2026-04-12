#pragma once

#include <opencv2/core.hpp>
#include <vector>

class PnpSolver {
public:
    PnpSolver();   // 构造函数内部硬编码所有参数

    // 可选：允许动态修改装甲板尺寸
    void setArmorSize(double width, double height);

    // 基础 PnP 解算
    bool solve(const std::vector<cv::Point2f>& image_points,
               cv::Mat& rvec,
               cv::Mat& tvec) const;

    // 增强版：同时返回欧拉角和距离
    bool solveWithPose(const std::vector<cv::Point2f>& image_points,
                       cv::Mat& rvec,
                       cv::Mat& tvec,
                       double& yaw,
                       double& pitch,
                       double& distance) const;

    cv::Mat getCamera_matrix_(){ return camera_matrix_; }

private:
    cv::Mat camera_matrix_;          // 相机内参矩阵
    cv::Mat dist_coeffs_;            // 畸变系数
    std::vector<cv::Point3f> object_points_; // 装甲板世界坐标（以中心为原点）
};