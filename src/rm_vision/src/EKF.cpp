#include "EKF.hpp"

EKF::EKF() : initialized_(false) {
    X_ = Eigen::VectorXd::Zero(10);
    P_ = Eigen::MatrixXd::Identity(10, 10) * 1.0;
    
    Q_ = Eigen::MatrixXd::Identity(10, 10);
    Q_.block<3,3>(0,0) *= 0.01; // XYZ位置噪声
    Q_.block<3,3>(3,3) *= 0.1;  // XYZ速度噪声
    Q_(6,6) = 0.05; Q_(7,7) = 0.5; // Yaw及角速度噪声
    Q_(8,8) = 0.05; Q_(9,9) = 0.5; // Pitch及角速度噪声

    R_ = Eigen::MatrixXd::Identity(5, 5);
    R_.block<3,3>(0,0) *= 0.05; // XYZ测量噪声
    R_(3,3) = 0.1;             // Yaw测量噪声
    R_(4,4) = 0.1;             // Pitch测量噪声
}

void EKF::init(const Eigen::Vector3d& p_armor, double yaw_abs, double pitch_abs) {
    // 根据 3D 球面模型，从装甲板位置反推中心位置
    double xc = p_armor.x() - r_ * cos(yaw_abs) * cos(pitch_abs);
    double yc = p_armor.y() - r_ * sin(yaw_abs) * cos(pitch_abs);
    double zc = p_armor.z() - r_ * sin(pitch_abs);
    
    X_ << xc, yc, zc, 0, 0, 0, yaw_abs, 0, pitch_abs, 0;
    initialized_ = true;
}

void EKF::predict(double dt) {
    if (!initialized_) return;
    
    // 状态转移 f(X)
    X_(0) += X_(3) * dt; X_(1) += X_(4) * dt; X_(2) += X_(5) * dt;
    X_(6) += X_(7) * dt; X_(6) = normalizeAngle(X_(6));
    X_(8) += X_(9) * dt; X_(8) = normalizeAngle(X_(8));

    // 雅可比 F
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(10, 10);
    F(0, 3) = F(1, 4) = F(2, 5) = F(6, 7) = F(8, 9) = dt;
    
    P_ = F * P_ * F.transpose() + Q_;
}

void EKF::update(const Eigen::Vector3d& p_armor, double yaw_abs, double pitch_abs) {
    // 1. 寻找 4 块板中最匹配的那块 (基于 Yaw)
    double min_diff = 1e9; int best_i = 0;
    for (int i = 0; i < 4; i++) {
        double t_yaw = X_(6) + i * M_PI / 2.0;
        double diff = std::abs(normalizeAngle(yaw_abs - t_yaw));
        if (diff < min_diff) { min_diff = diff; best_i = i; }
    }
    double matched_yaw = X_(6) + best_i * M_PI / 2.0;
    double matched_pitch = X_(8); // Pitch 不做四面体匹配

    // 2. 预测观测 h(X) (3D球面模型：X-Y 为水平面，Z 为高度)
    Eigen::VectorXd z_pred(5);
    z_pred << X_(0) + r_ * cos(matched_yaw) * cos(matched_pitch),
              X_(1) + r_ * sin(matched_yaw) * cos(matched_pitch),
              X_(2) + r_ * sin(matched_pitch),
              matched_yaw,
              matched_pitch;

    // 3. 计算雅可比 H (5x10)
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(5, 10);
    H(0, 0) = 1.0; 
    H(0, 6) = -r_ * sin(matched_yaw) * cos(matched_pitch);
    H(0, 8) = -r_ * cos(matched_yaw) * sin(matched_pitch);
    
    H(1, 1) = 1.0;
    H(1, 6) =  r_ * cos(matched_yaw) * cos(matched_pitch);
    H(1, 8) = -r_ * sin(matched_yaw) * sin(matched_pitch);
    
    H(2, 2) = 1.0; 
    H(2, 8) =  r_ * cos(matched_pitch);
    
    H(3, 6) = 1.0;
    H(4, 8) = 1.0;

    // 4. 更新
    Eigen::VectorXd z_meas(5);
    z_meas << p_armor.x(), p_armor.y(), p_armor.z(), yaw_abs, pitch_abs;
    
    Eigen::VectorXd y = z_meas - z_pred;
    y(3) = normalizeAngle(y(3));
    y(4) = normalizeAngle(y(4));

    Eigen::MatrixXd S = H * P_ * H.transpose() + R_;
    Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();
    
    X_ = X_ + K * y;
    X_(6) = normalizeAngle(X_(6));
    X_(8) = normalizeAngle(X_(8));
    
    P_ = (Eigen::MatrixXd::Identity(10, 10) - K * H) * P_;
}

double EKF::normalizeAngle(double angle) {
    static const double PI2 = 2.0 * M_PI;
    angle = fmod(angle + M_PI, PI2);
    if (angle < 0) angle += PI2;
    return angle - M_PI;
}

void EKF::getState(Eigen::Vector3d& pos_c, double& yaw, double& v_yaw, double& pitch, double& v_pitch) const {
    pos_c = X_.head<3>(); 
    yaw = X_(6); 
    v_yaw = X_(7);
    pitch = X_(8);
    v_pitch = X_(9);
}