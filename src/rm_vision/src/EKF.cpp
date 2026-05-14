#include "EKF.hpp"

EKF::EKF() : initialized_(false)
{
    X_ = Eigen::VectorXd::Zero(13);
    P_ = Eigen::MatrixXd::Identity(13, 13) * 100.0;

    // 过程噪声 Q (13x13)
    Q_ = Eigen::MatrixXd::Zero(13, 13);
    Q_(IDX_X, IDX_X) = 0.001; Q_(IDX_Y, IDX_Y) = 0.001; Q_(IDX_Z, IDX_Z) = 0.001;
    Q_(IDX_VX, IDX_VX) = 0.01; Q_(IDX_VY, IDX_VY) = 0.01; Q_(IDX_VZ, IDX_VZ) = 0.01;
    Q_(IDX_AX, IDX_AX) = 0.5;  Q_(IDX_AY, IDX_AY) = 0.5;  Q_(IDX_AZ, IDX_AZ) = 0.5;
    Q_(IDX_YAW,   IDX_YAW)   = 0.01;
    Q_(IDX_VYAW,  IDX_VYAW)  = 0.1;
    Q_(IDX_AYAW,  IDX_AYAW)  = 1.0;
    Q_(IDX_R, IDX_R) = 1e-6;   // 半径近乎常数

    // 测量噪声 R (4x4)
    R_ = Eigen::MatrixXd::Zero(4, 4);
    R_(0,0) = 0.05;   // x (m²)
    R_(1,1) = 0.05;   // y
    R_(2,2) = 0.10;   // z
    R_(3,3) = 0.02;   // yaw (rad²)
}

void EKF::init(const Eigen::Vector3d& p_armor, double yaw_abs)
{
    double r0 = 0.25;   // 初始猜测半径

    // 反推车体中心 (世界系)
    double xc = p_armor.x() - r0 * std::sin(yaw_abs);
    double yc = p_armor.y();                         // 假设 y 无横向偏移
    double zc = p_armor.z() - r0 * std::cos(yaw_abs);

    X_.setZero();
    X_(IDX_X)   = xc;
    X_(IDX_Y)   = yc;
    X_(IDX_Z)   = zc;
    X_(IDX_VX)  = 0.0;
    X_(IDX_VY)  = 0.0;
    X_(IDX_VZ)  = 0.0;
    X_(IDX_AX)  = 0.0;
    X_(IDX_AY)  = 0.0;
    X_(IDX_AZ)  = 0.0;
    X_(IDX_YAW)  = yaw_abs;
    X_(IDX_VYAW) = 0.0;
    X_(IDX_AYAW) = 0.0;
    X_(IDX_R)    = r0;

    P_ = Eigen::MatrixXd::Identity(13, 13) * 100.0;
    initialized_ = true;
}

void EKF::predict(double dt)
{
    if (!initialized_ || dt <= 0.0) return;

    // 匀加速状态预测
    X_(IDX_X) += X_(IDX_VX) * dt + 0.5 * X_(IDX_AX) * dt * dt;
    X_(IDX_Y) += X_(IDX_VY) * dt + 0.5 * X_(IDX_AY) * dt * dt;
    X_(IDX_Z) += X_(IDX_VZ) * dt + 0.5 * X_(IDX_AZ) * dt * dt;
    X_(IDX_VX) += X_(IDX_AX) * dt;
    X_(IDX_VY) += X_(IDX_AY) * dt;
    X_(IDX_VZ) += X_(IDX_AZ) * dt;
    X_(IDX_YAW)  += X_(IDX_VYAW) * dt + 0.5 * X_(IDX_AYAW) * dt * dt;
    X_(IDX_VYAW) += X_(IDX_AYAW) * dt;
    X_(IDX_YAW) = normalizeAngle(X_(IDX_YAW));
    // 半径保持不变

    // 状态转移雅可比 F (13x13)
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(13, 13);
    F(IDX_X,   IDX_VX) = dt;   F(IDX_X,   IDX_AX) = 0.5 * dt * dt;
    F(IDX_Y,   IDX_VY) = dt;   F(IDX_Y,   IDX_AY) = 0.5 * dt * dt;
    F(IDX_Z,   IDX_VZ) = dt;   F(IDX_Z,   IDX_AZ) = 0.5 * dt * dt;
    F(IDX_VX,  IDX_AX) = dt;
    F(IDX_VY,  IDX_AY) = dt;
    F(IDX_VZ,  IDX_AZ) = dt;
    F(IDX_YAW, IDX_VYAW) = dt; F(IDX_YAW, IDX_AYAW) = 0.5 * dt * dt;
    F(IDX_VYAW,IDX_AYAW) = dt;

    P_ = F * P_ * F.transpose() + Q_;
}

double EKF::normalizeAngle(double angle) const
{
    double a = std::fmod(angle, 2.0 * M_PI);
    if (a > M_PI)        a -= 2.0 * M_PI;
    else if (a < -M_PI)  a += 2.0 * M_PI;
    return a;
}

void EKF::update(const Eigen::Vector3d& p_armor, double yaw_abs)
{
    if (!initialized_) return;

    // 数据关联：选择最匹配的装甲板面
    double yaw_est = X_(IDX_YAW);
    double min_diff = 1e9;
    int best_idx = 0;
    for (int i = 0; i < 4; ++i) {
        double target_yaw = yaw_est + i * (M_PI / 2.0);
        double diff = std::abs(normalizeAngle(yaw_abs - target_yaw));
        if (diff < min_diff) {
            min_diff = diff;
            best_idx = i;
        }
    }
    double matched_yaw = yaw_est + best_idx * (M_PI / 2.0);

    double r_est = X_(IDX_R);

    // 预测观测 h(X)
    Eigen::VectorXd z_pred(4);
    z_pred(0) = X_(IDX_X) + r_est * std::sin(matched_yaw);
    z_pred(1) = X_(IDX_Y);   // y 无偏移
    z_pred(2) = X_(IDX_Z) + r_est * std::cos(matched_yaw);
    z_pred(3) = normalizeAngle(matched_yaw);

    // 观测雅可比 H (4x13)
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, 13);
    H(0, IDX_X)   = 1.0;
    H(0, IDX_YAW) =  r_est * std::cos(matched_yaw);
    H(0, IDX_R)   = std::sin(matched_yaw);

    H(1, IDX_Y)   = 1.0;

    H(2, IDX_Z)   = 1.0;
    H(2, IDX_YAW) = -r_est * std::sin(matched_yaw);
    H(2, IDX_R)   = std::cos(matched_yaw);

    H(3, IDX_YAW) = 1.0;

    // 测量向量
    Eigen::VectorXd z_meas(4);
    z_meas << p_armor.x(), p_armor.y(), p_armor.z(), yaw_abs;

    // 新息
    Eigen::VectorXd y_res = z_meas - z_pred;
    y_res(3) = normalizeAngle(y_res(3));

    // 卡尔曼更新
    Eigen::MatrixXd S = H * P_ * H.transpose() + R_;
    Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();
    X_ = X_ + K * y_res;
    P_ = (Eigen::MatrixXd::Identity(13, 13) - K * H) * P_;

    X_(IDX_YAW) = normalizeAngle(X_(IDX_YAW));
    if (X_(IDX_R) < 0.05) X_(IDX_R) = 0.05;   // 半径物理下限
}

void EKF::getState(Eigen::Vector3d& pos_c, double& yaw, double& v_yaw, double& r) const
{
    pos_c = X_.segment<3>(IDX_X);
    yaw   = X_(IDX_YAW);
    v_yaw = X_(IDX_VYAW);
    r     = X_(IDX_R);
}