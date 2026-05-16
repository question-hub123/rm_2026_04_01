#include "EKF.hpp"

EKF::EKF() : initialized_(false) {
    x = Eigen::Matrix<double, 9, 1>::Zero();
    P = Eigen::Matrix<double, 9, 9>::Identity() * 0.1;
    F = Eigen::Matrix<double, 9, 9>::Identity();
    // 观测矩阵 H 的固定部分: 观测为 [xa, ya, za, yaw]
    H.setZero();
    H(0,0) = 1.0; // xa = xc - r*cos(yaw) -> ∂xa/∂xc = 1
    H(1,2) = 1.0; // ya = yc - r*sin(yaw) -> ∂ya/∂yc = 1
    H(2,4) = 1.0; // za
    H(3,6) = 1.0; // yaw
    // H(0,6), H(0,8), H(1,6), H(1,8) 将在 update 中动态更新

    // 过程噪声协方差
    Q.setIdentity();
    Q.diagonal() << 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 1e-6;

    // 测量噪声协方差 (初始值，可根据实际情况调整)
    R.setIdentity();
    R.diagonal() << 0.05, 0.05, 0.05, 0.01;
}

void EKF::init(const Eigen::Vector3d& p_armor, double armor_yaw, double r0) {
    x.setZero();
    x(0) = p_armor.x() + r0 * cos(armor_yaw);
    x(2) = p_armor.y() + r0 * sin(armor_yaw);
    x(4) = p_armor.z();
    x(6) = armor_yaw;
    x(8) = r0;
    initialized_ = true;
}

void EKF::predict(double dt) {
    if (!initialized_ || dt <= 0.0) return;

    // 状态转移: 匀速模型 + 半径不变
    x(0) += x(1) * dt;
    x(2) += x(3) * dt;
    x(4) += x(5) * dt;
    x(6) += x(7) * dt; // yaw 连续变化，不归一化
    // 半径 r 不变
    // 速度 v 和角速度 v_yaw 保持不变 (假设加速度为0)

    // 更新状态转移矩阵 F
    F.setIdentity();
    F(0,1) = dt;
    F(2,3) = dt;
    F(4,5) = dt;
    F(6,7) = dt;

    P = F * P * F.transpose() + Q;
}

void EKF::update(const Eigen::Vector4d& z) {
    if (!initialized_) return;

    double yaw = x(6);
    double r = x(8);

    // 更新观测雅可比 H 中与 yaw, r 相关的元素
    H(0,6) = r * sin(yaw);   // ∂xa/∂yaw = r*sin(yaw)
    H(0,8) = -cos(yaw);      // ∂xa/∂r = -cos(yaw)
    H(1,6) = -r * cos(yaw);  // ∂ya/∂yaw = -r*cos(yaw)
    H(1,8) = -sin(yaw);      // ∂ya/∂r = -sin(yaw)

    // 预测观测
    Eigen::Vector4d z_pred;
    z_pred(0) = x(0) - r * cos(yaw); // xa
    z_pred(1) = x(2) - r * sin(yaw); // ya
    z_pred(2) = x(4);                // za
    z_pred(3) = yaw;                 // yaw_continuous

    // 卡尔曼更新
    Eigen::Matrix4d S = H * P * H.transpose() + R;
    Eigen::Matrix<double, 9, 4> K = P * H.transpose() * S.inverse();
    Eigen::Vector4d y = z - z_pred;
    // 角度残差归一化到 [-pi, pi]，因为我们使用连续化yaw，这里不能直接归一化到-pi~pi，
    // 而应使用最短角度距离。我们的观测z(3)已经是连续化后的yaw，x(6)也是连续化的，
    // 所以残差直接相减就是最短距离（因为两者都在同一连续域）。
    // 但为防止数值跳变，可以调用 normalizeAngle 处理，但连续化后的值可能超出 [-pi,pi]，
    // 所以不做归一化，直接相减即可。
    // 如果遇到第一次初始化时未连续化的情形，残差可能很大，但这无妨。
    x += K * y;
    // 注意：x(6) 保持连续化，不做归一化
    P = (Eigen::Matrix<double, 9, 9>::Identity() - K * H) * P;
}

Eigen::Vector3d EKF::getArmorPosition() const {
    double yaw = x(6);
    double r = x(8);
    return Eigen::Vector3d(x(0) - r * cos(yaw), x(2) - r * sin(yaw), x(4));
}

Eigen::Vector3d EKF::getVehiclePosition() const {
    return Eigen::Vector3d(x(0), x(2), x(4));
}

double EKF::getRadius() const { return x(8); }

double EKF::getContinuousYaw() const { return x(6); }

double EKF::normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;
    return angle;
}

double EKF::shortestAngularDistance(double from, double to) {
    return normalizeAngle(to - from);
}