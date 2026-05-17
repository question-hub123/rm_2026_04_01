#pragma once
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <functional>

class EKF {
public:
    // 10维状态向量: [xc, vxc, yc, vyc, zc, vzc, yaw, v_yaw, r, zp]^T
    // xc, yc, zc: 机器人中心在世界坐标系下的位置
    // vxc, vyc, vzc: 机器人中心的速度
    // yaw: 正在追踪的装甲板朝向角 (连续化)
    // v_yaw: 装甲板自转角速度
    // r: 旋转半径
    // zp: 当前装甲板高度相对于车体中心的高度偏移量 (z_armor = zc + zp)
    Eigen::Matrix<double, 10, 1> x;
    Eigen::Matrix<double, 10, 10> P;
    Eigen::Matrix<double, 10, 10> F;
    Eigen::Matrix<double, 4, 10> H;
    Eigen::Matrix<double, 10, 10> Q;
    Eigen::Matrix<double, 4, 4> R;

    bool initialized_;

    EKF();

    // 初始化滤波器
    void init(const Eigen::Vector3d& p_armor, double armor_yaw, double r0 = 0.26, double zp0 = 0.0);

    // 状态预测
    void predict(double dt);

    // 观测更新 (输入量: 球面坐标下的 yaw, pitch, distance, 以及姿态角 orientation_yaw)
    void update(const Eigen::Vector4d& z);

    // 从当前状态反推装甲板在世界系下的坐标
    Eigen::Vector3d getArmorPosition() const;

    // 获取滤波后的车体中心位置
    Eigen::Vector3d getVehiclePosition() const;

    // 获取装甲板的相对中心高度差
    double getArmorZOffset() const;

    // 获取当前估计的半径
    double getRadius() const;

    // 获取连续化后的当前装甲板 yaw 
    double getContinuousYaw() const;

    // 辅助函数：角度归一化到 [-pi, pi]
    static double normalizeAngle(double angle);
    
    // 辅助函数：计算最短角距离
    static double shortestAngularDistance(double from, double to);

private:
    // 数值雅可比矩阵求解
    Eigen::MatrixXd numericalJacobian(const Eigen::VectorXd& x0,
        const std::function<Eigen::Vector4d(const Eigen::VectorXd&)>& h_func,
        double eps = 1e-5) const;
};