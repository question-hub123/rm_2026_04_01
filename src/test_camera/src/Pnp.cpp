#include "Pnp.hpp"
#include <opencv2/calib3d.hpp>
#include <cmath>
#include <opencv2/core/mat.hpp>

PnpSolver::PnpSolver()
{
    // 硬编码相机内参（根据你提供的参数）
    camera_matrix_ = (cv::Mat_<double>(3,3) <<
        2374.54248, 0.0,        698.85288,
        0.0,        2377.53648, 520.8649,
        0.0,        0.0,        1.0);

    // 硬编码畸变系数
    dist_coeffs_ = (cv::Mat_<double>(1,5) <<
        -0.059743, 0.355479, -0.000625, 0.001595, 0.000000);

    // 硬编码装甲板尺寸（米）
    double armor_width = 0.095;   // 95mm
    double armor_height = 0.095;  // 85mm
    setArmorSize(armor_width, armor_height);
}

void PnpSolver::setArmorSize(double width, double height)
{
    object_points_.clear();
    // 世界坐标顺序：左上、右上、右下、左下（与图像角点顺序一致）
    object_points_.emplace_back(-width/2.0,  height/2.0, 0.0); // 左上
    object_points_.emplace_back( width/2.0,  height/2.0, 0.0); // 右上
    object_points_.emplace_back( width/2.0, -height/2.0, 0.0); // 右下
    object_points_.emplace_back(-width/2.0, -height/2.0, 0.0); // 左下
}

bool PnpSolver::solve(const std::vector<cv::Point2f>& image_points,
                      cv::Mat& rvec,
                      cv::Mat& tvec) const
{
    if (image_points.size() != 4) return false;
    return cv::solvePnP(object_points_, image_points,
                        camera_matrix_, dist_coeffs_,
                        rvec, tvec, false, cv::SOLVEPNP_IPPE);
}

bool PnpSolver::solveWithPose(const std::vector<cv::Point2f>& image_points,
                              cv::Mat& rvec,
                              cv::Mat& tvec,
                              double& yaw,
                              double& pitch,
                              double& distance) const
{
    if (!solve(image_points, rvec, tvec)) return false;

    // 计算距离（米）
    distance = std::sqrt(tvec.at<double>(0)*tvec.at<double>(0) +
                         tvec.at<double>(1)*tvec.at<double>(1) +
                         tvec.at<double>(2)*tvec.at<double>(2));

    // 计算欧拉角（度）
    cv::Mat rot_mat;
    cv::Rodrigues(rvec, rot_mat);
    yaw   = std::atan2(rot_mat.at<double>(1,0), rot_mat.at<double>(0,0));
    pitch = std::asin(-rot_mat.at<double>(2,0));
    yaw   *= 180.0 / M_PI;
    pitch *= 180.0 / M_PI;

    return true;
}