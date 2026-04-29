
#include <Kalman.hpp>
#include <cmath>
#include <eigen3/Eigen/src/Core/Matrix.h>

KF::KF(double dt)
    : initialized_(false)
{
    // 状态转移矩阵 A（dt 稍后在 predict 中动态设置）
    A_ = Eigen::Matrix4d::Identity();
    // 观测矩阵 H：观测 yaw 和 pitch
    H_.setZero();
    H_(0, 0) = 1.0;   // yaw
    H_(1, 2) = 1.0;   // pitch

    // 过程噪声协方差 Q（需要根据实际情况调参）
    Q_ = Eigen::Matrix4d::Zero();
    Q_(0, 0) = 0.08;   // yaw 过程噪声
    Q_(1, 1) = 0.2;    // yaw_rate 过程噪声
    Q_(2, 2) = 0.08;   // pitch 过程噪声
    Q_(3, 3) = 0.2;    // pitch_rate 过程噪声

    // 测量噪声协方差 R（单位：弧度²，假设标准差约 0.03 rad ≈ 1.7°）
    R_ = Eigen::Matrix2d::Zero();
    R_(0, 0) = 1.0;  // yaw 测量噪声方差
    R_(1, 1) = 1.0;  // pitch 测量噪声方差

    // 初始协方差 P
    P_ = Eigen::Matrix4d::Identity() * 100.0;

    X_ = Eigen::VectorXd(4);
}

void KF::init(double yaw, double pitch)
{
    X_ << yaw, 0.0, pitch, 0.0;
    P_ = Eigen::Matrix4d::Identity() * 100.0;
    initialized_ = true;
}

void KF::predict(double dt)
{
    if (!initialized_) return;

    // 更新状态转移矩阵 A 中的 dt 项
    A_(0, 1) = dt;   // yaw += yaw_rate * dt
    A_(2, 3) = dt;   // pitch += pitch_rate * dt

    // 状态预测
    X_ = A_ * X_;
    // 协方差预测
    P_ = A_ * P_ * A_.transpose() + Q_;
}

double KF::normalizeAngle(double angle)
{
    double a = std::fmod(angle, 2.0 * M_PI);
    if (a > M_PI) a -= 2.0 * M_PI;
    if (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

void KF::update(double yaw_meas, double pitch_meas)
{
    if (!initialized_) return;

    // 观测向量
    Eigen::Vector2d Z;
    Z << yaw_meas, pitch_meas;

    // 预测观测
    Eigen::Vector2d Z_pred = H_ * X_;

    // 计算残差，并对角度分量归一化
    double yaw_res = normalizeAngle(Z(0) - Z_pred(0));
    double pitch_res = normalizeAngle(Z(1) - Z_pred(1));
    Eigen::Vector2d y;
    y << yaw_res, pitch_res;

    // 创新协方差 S = H * P * H^T + R
    Eigen::Matrix2d S = H_ * P_ * H_.transpose() + R_;
    // 卡尔曼增益 K = P * H^T * S^{-1}
    Eigen::Matrix<double, 4, 2> K = P_ * H_.transpose() * S.inverse();

    // 状态更新
    X_ = X_ + K * y;
    // 协方差更新
    P_ = (Eigen::Matrix4d::Identity() - K * H_) * P_;

    // 可选：对更新后的角度再次归一化（防止数值累积误差）
    X_(0) = normalizeAngle(X_(0));
    X_(2) = normalizeAngle(X_(2));
}

void KF::getState(double& yaw, double& yaw_rate, double& pitch, double& pitch_rate) const
{
    yaw = X_(0);
    yaw_rate = X_(1);
    pitch = X_(2);
    pitch_rate = X_(3);
}