#pragma once
#include <eigen3/Eigen/Dense>
#include <cmath>

class EKF {
public:
    EKF();

    // 初始化：传入观测到的装甲板世界坐标 (m) 和绝对偏航角 (rad)
    void init(const Eigen::Vector3d& p_armor, double yaw_abs);

    // 预测：匀加速模型
    void predict(double dt);

    // 更新：观测装甲板的世界坐标和绝对偏航角
    void update(const Eigen::Vector3d& p_armor, double yaw_abs);

    // 获取滤波后的车体状态
    void getState(Eigen::Vector3d& pos_c, double& yaw, double& v_yaw) const;

    // 加速度访问接口
    double getVx() const;
    double getVy() const;
    double getVz() const;
    double getAx() const;
    double getAy() const;
    double getAz() const;
    double getAYaw() const;

    bool isInitialized() const { return initialized_; }

private:
    bool initialized_;
    double r_ = 0.25;               // 车辆半径 (m)

    // 状态索引
    static constexpr int IDX_X = 0, IDX_Y = 1, IDX_Z = 2;
    static constexpr int IDX_VX = 3, IDX_VY = 4, IDX_VZ = 5;
    static constexpr int IDX_AX = 6, IDX_AY = 7, IDX_AZ = 8;
    static constexpr int IDX_YAW = 9, IDX_VYAW = 10, IDX_AYAW = 11;

    Eigen::VectorXd X_;             // 12 维状态
    Eigen::MatrixXd P_;             // 12x12 协方差
    Eigen::MatrixXd Q_;             // 12x12 过程噪声
    Eigen::MatrixXd R_;             // 4x4 测量噪声

    double normalizeAngle(double angle) const;
};