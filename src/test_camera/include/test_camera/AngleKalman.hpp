#pragma once
#include <eigen3/Eigen/Dense>
#include <cmath>

class AngleKalman
{
public:
    AngleKalman(double dt = 0.33);

    void init(double yaw, double pitch);

    void predict(double dt);

    void update(double yaw_meas, double pitch_measure);

    void getState(double& yaw, double& yaw_rate, double& pitch, double& pitch_rate) const;

    bool isInitialized() const { return initialized_; }

private:
    bool initialized_;

    // 状态向量 [yaw, yaw_rate, pitch, pitch_rate]ᵀ
    Eigen::Vector4d X_;
    // 状态协方差矩阵
    Eigen::Matrix4d P_;
    // 状态转移矩阵（动态，因为 dt 可变）
    Eigen::Matrix4d A_;
    // 观测矩阵（固定）
    Eigen::Matrix<double, 2, 4> H_;
    // 过程噪声协方差矩阵
    Eigen::Matrix4d Q_;
    // 测量噪声协方差矩阵
    Eigen::Matrix2d R_;

    // 角度归一化到 [-π, π]
    double normalizeAngle(double angle);
};