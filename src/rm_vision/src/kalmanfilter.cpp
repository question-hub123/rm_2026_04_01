
#include <Kalman.hpp>
#include <cmath>

KF::KF(double dt) : dt_(dt)
{
    X_ = Eigen::VectorXd::Zero(5);
    P_ = Eigen::MatrixXd::Identity(5,5) * 1000.0;

    A_ = Eigen::MatrixXd::Identity(5,5);
    A_(0,2) = dt_;
    A_(1,3) = dt_;

    H_ = Eigen::MatrixXd::Zero(3,5);
    H_(0,0) = 1;
    H_(1,1) = 1;
    H_(2, 4) = 1;

    Q_ = Eigen::MatrixXd::Zero(5, 5);
    Q_(0,0) = 1.0; Q_(1,1) = 1.0;   // 位置过程噪声
    Q_(2,2) = 10.0; Q_(3,3) = 10.0; // 速度过程噪声
    Q_(4,4) = 0.1;                  // 角度过程噪声

    R_ = Eigen::MatrixXd::Zero(3, 3);
    R_(0,0) = 25.0;   // x 测量噪声方差（mm²）
    R_(1,1) = 25.0;   // y 测量噪声方差（mm²）
    R_(2,2) = 1.0;    // yaw 测量噪声方差（度²)
}

void KF::init(double x, double y, double yaw) 
{
    X_ << x, y, 0.0, 0.0, yaw;
    P_ = Eigen::MatrixXd::Identity(5, 5) * 1000.0;
    initialized_ = true;
}

void KF::predict(double dt) 
{
    if (!initialized_) return;
    A_(0,2) = dt;
    A_(1,3) = dt;

    X_ = A_ * X_;
    P_ = A_ * P_ * A_.transpose() + Q_;
}

void KF::update(double zx, double zy, double zyaw) 
{
    if (!initialized_) return;

    Eigen::VectorXd Z(3);
    Z << zx, zy, zyaw;

    // 卡尔曼增益
    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
    Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();

    // 更新状态和协方差
    X_ = X_ + K * (Z - H_ * X_);
    P_ = (Eigen::MatrixXd::Identity(5, 5) - K * H_) * P_;
}

void KF::getState(double& x, double& y, double& vx, double& vy, double& yaw) const 
{
    x = X_(0);
    y = X_(1);
    vx = X_(2);
    vy = X_(3);
    yaw = X_(4);
}