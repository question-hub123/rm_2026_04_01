#pragma once
#include <eigen3/Eigen/Dense>
#include <cmath>

class EKF {
public:
    EKF();
    // 初始化：传入观测到的装甲板世界坐标、绝对偏航角、绝对俯仰角
    void init(const Eigen::Vector3d& p_armor, double yaw_abs, double pitch_abs);
    // 预测：匀速模型
    void predict(double dt);
    // 更新：观测装甲板的世界坐标、绝对偏航角、绝对俯仰角
    void update(const Eigen::Vector3d& p_armor, double yaw_abs, double pitch_abs);
    // 获取预测状态：车体中心、偏航角及速度、俯仰角及速度
    void getState(Eigen::Vector3d& pos_c, double& yaw, double& v_yaw, double& pitch, double& v_pitch) const;
    
    bool isInitialized() const { return initialized_; }

private:
    bool initialized_;
    double r_ = 0.25; // 目标车辆半径

    Eigen::VectorXd X_; // [xc, yc, zc, vx, vy, vz, yaw, v_yaw, pitch, v_pitch]
    Eigen::MatrixXd P_; // 协方差 (10x10)
    Eigen::MatrixXd Q_; // 过程噪声 (10x10)
    Eigen::MatrixXd R_; // 测量噪声 (5x5)

    double normalizeAngle(double angle);
};