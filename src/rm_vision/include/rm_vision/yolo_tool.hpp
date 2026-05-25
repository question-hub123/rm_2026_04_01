#pragma once
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

class Yolo_Tool
{
public:
    // 输入四个点，输出按 [左上, 右上, 右下, 左下] 顺序排列的 vector
    std::vector<cv::Point2f> orderPoints(const std::vector<cv::Point2f>& pts) 
    {
        cv::Point2f center(0, 0);
        for (const auto& p : pts) {
            center += p;
        }
        center *= (1.0 / pts.size());

        std::vector<std::pair<float, int>> angles;
        for (int i = 0; i < pts.size(); ++i) {
            float angle = std::atan2(pts[i].y - center.y, pts[i].x - center.x);
            angles.emplace_back(angle, i);
        }

        std::sort(angles.begin(), angles.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

        std::vector<cv::Point2f> ordered;
        for (auto& idx : angles) {
            ordered.push_back(pts[idx.second]);
        }

        auto minSum = ordered[0].x + ordered[0].y;
        int startIdx = 0;
        for (int i = 1; i < 4; ++i) {
            float sum = ordered[i].x + ordered[i].y;
            if (sum < minSum) {
                minSum = sum;
                startIdx = i;
            }
        }

        std::vector<cv::Point2f> result;
        for (int i = 0; i < 4; ++i) {
            result.push_back(ordered[(startIdx + i) % 4]);
        }

        return result;
    }

    Eigen::Matrix3d get_R_gimbal2world(double gimbal_yaw, double gimbal_pitch) const
    {
        Eigen::Matrix3d R_gimbal2world;
        R_gimbal2world = Eigen::AngleAxisd(gimbal_yaw,   Eigen::Vector3d::UnitZ())
               * Eigen::AngleAxisd(gimbal_pitch, Eigen::Vector3d::UnitY())
               * Eigen::AngleAxisd(0.0,          Eigen::Vector3d::UnitX());
        return R_gimbal2world;
    }
    

    void drawVehicleCenter(cv::Mat& img, const Eigen::Vector3d& center_world, const Eigen::Quaterniond& q_imu) const 
    {
        // 标定矩阵（与 cameraToWorld 新版本完全一致）
        Eigen::Matrix3d R_camera2gimbal;
        R_camera2gimbal << 0,  0,  1,
                        -1,  0,  0,
                        0, -1,  0;
        Eigen::Vector3d t_camera2gimbal(0, 0, 0);   // 可后续标定填充
        Eigen::Matrix3d R_gimbal2imubody = Eigen::Matrix3d::Identity();

        // 1. 世界 → 云台
        Eigen::Matrix3d R_imu2world = q_imu.toRotationMatrix();
        Eigen::Matrix3d R_gimbal2world = R_imu2world * R_gimbal2imubody.transpose();
        Eigen::Vector3d center_gimbal = R_gimbal2world.transpose() * center_world;

        // 2. 云台 → 相机（手眼标定逆变换）
        Eigen::Vector3d center_cam = R_camera2gimbal.transpose() * (center_gimbal - t_camera2gimbal);

        // 3. 相机内参投影
        cv::Mat K = (cv::Mat_<double>(3,3) <<
            1296.16167, 0.0,        643.60901,
            0.0,        1296.23028, 509.49319,
            0.0,        0.0,        1.0);

        if (center_cam.z() <= 0.1) return;

        double fx = K.at<double>(0,0);
        double fy = K.at<double>(1,1);
        double cx = K.at<double>(0,2);
        double cy = K.at<double>(1,2);
        double u = fx * center_cam.x() / center_cam.z() + cx;
        double v = fy * center_cam.y() / center_cam.z() + cy;

        cv::Point2f px(static_cast<float>(u), static_cast<float>(v));
        cv::circle(img, px, 8, cv::Scalar(0, 0, 255), -1);
        cv::putText(img, "CarCtr", px + cv::Point2f(10, -10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    }

    
    /**
     * @brief 根据滤波后的车体中心和车体朝向，重投影绘制 4 块装甲板
     * @param img 画面
     * @param center_world EKF算出的车体中心坐标
     * @param vehicle_yaw EKF算出的车体偏航角(连续yaw)
     * @param q_imu IMU四元数
     * @param radius 旋转半径(默认0.26)
     */
    /**
     * @brief 根据滤波后的车体中心和车体朝向，重投影绘制 4 块装甲板
     */
    void drawAllArmors(cv::Mat& img, 
                       const Eigen::Vector3d& center_world, 
                       double vehicle_yaw, 
                       const Eigen::Quaterniond& q_imu, 
                       double radius = 0.26) const 
    {
        const double ARMOR_W = 0.135;
        const double ARMOR_H = 0.055;

        const cv::Mat K = (cv::Mat_<double>(3,3) <<
            1296.16167, 0.0,        643.60901,
            0.0,        1296.23028, 509.49319,
            0.0,        0.0,        1.0);

        Eigen::Matrix3d R_camera2gimbal;
        R_camera2gimbal << 0,  0,  1,
                        -1,  0,  0,
                         0, -1,  0;

        Eigen::Matrix3d R_gimbal2world = q_imu.toRotationMatrix();
        Eigen::Matrix3d R_world2gimbal = R_gimbal2world.transpose();
        Eigen::Matrix3d R_gimbal2camera = R_camera2gimbal.transpose();

        for (int i = 0; i < 4; ++i) {
            // face_yaw 是第 i 块装甲板的【向内法向】
            double face_yaw = vehicle_yaw + i * (M_PI / 2.0);

            // 1. 装甲板中心 = 车体中心 - r * 向内法向
            Eigen::Vector3d board_center_world;
            board_center_world.x() = center_world.x() - radius * std::cos(face_yaw);
            board_center_world.y() = center_world.y() - radius * std::sin(face_yaw);
            board_center_world.z() = center_world.z();

            // 2. 计算装甲板的右向和上向向量
            // 因为向内法向是 (cos, sin, 0)，那么向外法向就是 (-cos, -sin, 0)
            // 右向向量 = 向外法向 叉乘 上方向(0,0,1) = (-sin, cos, 0)
            Eigen::Vector3d right_vec(-std::sin(face_yaw) * (ARMOR_W / 2.0), 
                                       std::cos(face_yaw) * (ARMOR_W / 2.0), 
                                       0.0);
            Eigen::Vector3d up_vec(0.0, 0.0, ARMOR_H / 2.0);

            // 3. 计算四个角点
            std::vector<Eigen::Vector3d> corners_world = {
                board_center_world - right_vec + up_vec, // 左上
                board_center_world + right_vec + up_vec, // 右上
                board_center_world + right_vec - up_vec, // 右下
                board_center_world - right_vec - up_vec  // 左下
            };

            // 4. 投影到像素坐标
            std::vector<cv::Point2f> img_pts;
            bool valid = true;
            for (const auto& pt_w : corners_world) {
                Eigen::Vector3d pt_c = R_gimbal2camera * (R_world2gimbal * pt_w);
                if (pt_c.z() <= 0.1) {
                    valid = false;
                    break;
                }
                float u = K.at<double>(0,0) * pt_c.x() / pt_c.z() + K.at<double>(0,2);
                float v = K.at<double>(1,1) * pt_c.y() / pt_c.z() + K.at<double>(1,2);
                img_pts.push_back(cv::Point2f(u, v));
            }

            // 5. 绘制
            if (valid) {
                // 主目标画红色，其余画青色
                cv::Scalar color = (i == 0) ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 0);
                for (int j = 0; j < 4; ++j) {
                    cv::line(img, img_pts[j], img_pts[(j+1)%4], color, 2, cv::LINE_AA);
                }
            }
        }
    }

    // ==================== 新增接口（基于四元数） ====================
    /**
    * @brief 相机坐标系 → 世界坐标系（使用 IMU 四元数 + 临时标定矩阵）
    */
    Eigen::Vector3d cameraToWorld(const Eigen::Vector3d& pos_cam, const Eigen::Quaterniond& q_imu) const {
        Eigen::Matrix3d R_camera2gimbal;
        R_camera2gimbal << 0,  0,  1,
                        -1,  0,  0,
                         0, -1,  0;
        Eigen::Vector3d t_camera2gimbal(0, 0, 0);
        Eigen::Matrix3d R_gimbal2imubody = Eigen::Matrix3d::Identity();

        Eigen::Vector3d p_gimbal = R_camera2gimbal * pos_cam + t_camera2gimbal;

        Eigen::Matrix3d R_imu2world = q_imu.toRotationMatrix();
        Eigen::Matrix3d R_gimbal2world = R_imu2world * R_gimbal2imubody.transpose();
        return R_gimbal2world * p_gimbal;
    }

    /**
    * @brief 将装甲板在相机系下的姿态（rvec）转换为世界系下的欧拉角（yaw, pitch, roll）
    */
    void cameraNormalToWorld(const cv::Mat& rvec,
                            const Eigen::Quaterniond& q_imu,
                            double& world_yaw,
                            double& world_pitch,
                            double& world_roll) const {
        Eigen::Matrix3d R_camera2gimbal;
        R_camera2gimbal << 0,  0,  1,
                        -1,  0,  0,
                         0, -1,  0;
        Eigen::Matrix3d R_gimbal2imubody = Eigen::Matrix3d::Identity();

        cv::Mat rmat;
        cv::Rodrigues(rvec, rmat);
        Eigen::Matrix3d R_armor2camera;
        // 手动将 cv::Mat 转为 Eigen::Matrix3d
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                R_armor2camera(i, j) = rmat.at<double>(i, j);

        Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal * R_armor2camera;
        Eigen::Matrix3d R_imu2world = q_imu.toRotationMatrix();
        Eigen::Matrix3d R_gimbal2world = R_imu2world * R_gimbal2imubody.transpose();
        Eigen::Matrix3d R_armor2world = R_gimbal2world * R_armor2gimbal;

        Eigen::Vector3d normal_world = R_armor2world * Eigen::Vector3d(0, 0, -1);

        world_yaw   = std::atan2(normal_world.y(), normal_world.x());
        world_pitch = std::asin(normal_world.z()); // 原来我写的 -normal_world.z() 是多余的，恢复不带负号即可
        world_roll  = std::atan2(R_armor2world(2,1), R_armor2world(2,2));
    }
};