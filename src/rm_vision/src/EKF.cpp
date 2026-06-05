// ekf11.cpp
#include "EKF.hpp"
#include <cmath>
#include <algorithm>

EKF11::EKF11() : armor_num_(4), last_id_(0), update_count_(0), converged_(false) {
    x.setZero();
    P = Eigen::Matrix<double, 11, 11>::Identity() * 100.0;
}

void EKF11::init(const Eigen::Vector3d& p_armor_world,
                 double orientation_yaw,
                 int armor_num,
                 double init_r,
                 double init_l,
                 double init_h,
                 const Eigen::Matrix<double, 11, 1>& P0_diag)
{
    armor_num_ = armor_num;
    last_id_ = 0;
    update_count_ = 0;
    converged_ = false;

    // 从装甲板中心反算旋转中心
    double cx = p_armor_world(0) + init_r * std::cos(orientation_yaw);
    double cy = p_armor_world(1) + init_r * std::sin(orientation_yaw);
    double cz = p_armor_world(2);

    x << cx, 0.0, cy, 0.0, cz, 0.0, orientation_yaw, 0.0, init_r, init_l, init_h;
    P = P0_diag.asDiagonal();
}

void EKF11::predict(double dt) {
    if (dt <= 0.0) return;

    // 状态转移矩阵 F (11x11) —— 匀速模型
    Eigen::Matrix<double, 11, 11> F = Eigen::Matrix<double, 11, 11>::Identity();
    F(0, 1) = dt;
    F(2, 3) = dt;
    F(4, 5) = dt;
    F(6, 7) = dt;

    // 分段白噪声过程噪声协方差 Q
    double v1 = 500.0;   // 加速度方差
    double v1_z = 0.01;
    double v2 = 400.0;    // 角加速度方差
    double a = dt * dt * dt * dt / 4.0;
    double b = dt * dt * dt / 2.0;
    double c = dt * dt;

    Eigen::Matrix<double, 11, 11> Q = Eigen::Matrix<double, 11, 11>::Zero();
    auto setBlock = [&](int i, double av) {
        Q(i, i)     = a * av;  Q(i, i+1)   = b * av;
        Q(i+1, i)   = b * av;  Q(i+1, i+1) = c * av;
    };
    setBlock(0, v1);  // x, vx
    setBlock(2, v1);  // y, vy
    setBlock(4, v1_z);  // z, vz
    setBlock(6, v2);  // yaw, vyaw
    // r, l, h 的噪声为 0

    // 状态预测（带角度归一化）
    Eigen::Matrix<double, 11, 1> x_pred = F * x;
    x_pred(6) = limitRad(x_pred(6));

    x = x_pred;
    P = F * P * F.transpose() + Q;
}

void EKF11::update(const Eigen::Vector4d& z_obs, const Eigen::Vector3d& armor_xyz) {
    // ---------- 装甲板 ID 匹配 ----------
    int best_id = 0;
    double min_error = 1e10;
    for (int i = 0; i < armor_num_; ++i) {
        double angle = limitRad(x(6) + i * 2.0 * M_PI / armor_num_);
        Eigen::Vector3d pred_xyz = hArmorXyz(x, i);
        Eigen::Vector3d pred_ypd = xyz2ypd(pred_xyz);
        double err = std::abs(limitRad(z_obs(3) - angle))
                   + std::abs(limitRad(z_obs(0) - pred_ypd(0)));
        if (err < min_error) 
        {
            min_error = err;
            best_id = i;
        }
    }
    last_id_ = best_id;
    update_count_++;

    // ---------- 观测噪声 R (动态) ----------
    double center_yaw = std::atan2(armor_xyz(1), armor_xyz(0));
    double delta_angle = limitRad(z_obs(3) - center_yaw);
    double dist = armor_xyz.norm();
    Eigen::Matrix<double, 4, 4> R = Eigen::Matrix<double, 4, 4>::Zero();
    R(0,0) = 2e-3;
    R(1,1) = 2e-3;
    R(2,2) = std::log(std::abs(delta_angle) + 1.5) + 1.0;
    R(3,3) = std::log(dist + 1.0) / 200.0 + 9e-2;

    // ---------- 解析雅可比 H ----------
    Eigen::Matrix<double, 4, 11> H = h_jacobian(x, best_id);

    // ---------- 观测残差 ----------
    Eigen::Vector4d z_pred = h_func(x, best_id);
    Eigen::Vector4d y = z_obs - z_pred;
    y(0) = limitRad(y(0));
    y(1) = limitRad(y(1));
    y(3) = limitRad(y(3));

    // ---------- EKF 更新 ----------
    Eigen::Matrix4d S = H * P * H.transpose() + R;
    Eigen::Matrix<double, 11, 4> K = P * H.transpose() * S.inverse();
    Eigen::Matrix<double, 11, 1> dx = K * y;
    x += dx;
    x(6) = limitRad(x(6));   // 角度归一化
    Eigen::Matrix<double, 11, 11> I = Eigen::Matrix<double, 11, 11>::Identity();
    P = (I - K * H) * P;

    // 收敛判断（简化）
    if (update_count_ > 5 && x(8) > 0.05 && x(8) < 0.5)
        converged_ = true;

}

Eigen::Vector3d EKF11::getArmorCenter(int id) const {
    return hArmorXyz(x, id);
}

Eigen::Vector3d EKF11::getArmorCenter() const {
    return hArmorXyz(x, last_id_);
}

// ==================== 私有工具函数 ====================

Eigen::Vector3d EKF11::hArmorXyz(const Eigen::Matrix<double, 11, 1>& state, int id) const {
    double angle = limitRad(state(6) + id * 2.0 * M_PI / armor_num_);
    bool use_long = (armor_num_ == 4) && (id == 1 || id == 3);
    double r = use_long ? (state(8) + state(9)) : state(8);
    double z = use_long ? (state(4) + state(10)) : state(4);
    double ax = state(0) - r * std::cos(angle);
    double ay = state(2) - r * std::sin(angle);
    return {ax, ay, z};
}

Eigen::Vector4d EKF11::h_func(const Eigen::Matrix<double, 11, 1>& state, int id) const {
    Eigen::Vector3d axyz = hArmorXyz(state, id);
    Eigen::Vector3d ypd = xyz2ypd(axyz);
    double angle = limitRad(state(6) + id * 2.0 * M_PI / armor_num_);
    return {ypd(0), ypd(1), ypd(2), angle};
}

Eigen::Matrix<double, 4, 11> EKF11::h_jacobian(const Eigen::Matrix<double, 11, 1>& state, int id) const 
{
    double angle = limitRad(state(6) + id * 2.0 * M_PI / armor_num_);
    bool use_long = (armor_num_ == 4) && (id == 1 || id == 3);
    double r = use_long ? (state(8) + state(9)) : state(8);

    // 对装甲板中心坐标的雅可比 (3x11)
    Eigen::Matrix<double, 3, 11> H_xyz = Eigen::Matrix<double, 3, 11>::Zero();
    H_xyz(0,0) = 1.0;
    H_xyz(0,6) =  r * std::sin(angle);
    H_xyz(0,8) = -std::cos(angle);
    if (use_long) H_xyz(0,9) = -std::cos(angle);

    H_xyz(1,2) = 1.0;
    H_xyz(1,6) = -r * std::cos(angle);
    H_xyz(1,8) = -std::sin(angle);
    if (use_long) H_xyz(1,9) = -std::sin(angle);

    H_xyz(2,4) = 1.0;
    if (use_long) H_xyz(2,10) = 1.0;

    // 球坐标转换雅可比
    Eigen::Vector3d axyz = hArmorXyz(state, id);
    Eigen::Matrix<double, 3, 3> J_ypd = xyz2ypdJacobian(axyz);

    // 组合成最终观测雅可比 4x11
    Eigen::Matrix<double, 4, 11> H = Eigen::Matrix<double, 4, 11>::Zero();
    H.block<3,11>(0,0) = J_ypd * H_xyz;
    H(3,6) = 1.0;   // 角度直接对应 yaw
    return H;
}

// ================== 静态工具函数 ==================

double EKF11::limitRad(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

Eigen::Vector3d EKF11::xyz2ypd(const Eigen::Vector3d& xyz) {
    double yaw = std::atan2(xyz(1), xyz(0));
    double pitch = std::atan2(xyz(2), std::sqrt(xyz(0)*xyz(0) + xyz(1)*xyz(1)));
    double dist = xyz.norm();
    return {yaw, pitch, dist};
}

Eigen::Matrix<double, 3, 3> EKF11::xyz2ypdJacobian(const Eigen::Vector3d& xyz) {
    double x = xyz(0), y = xyz(1), z = xyz(2);
    double dxy2 = x*x + y*y;
    double dxy = std::sqrt(dxy2);
    double d = xyz.norm();
    double d2 = d*d;

    Eigen::Matrix<double, 3, 3> J;
    J(0,0) = -y / dxy2;
    J(0,1) =  x / dxy2;
    J(0,2) = 0.0;

    J(1,0) = -x*z / (d2 * dxy);
    J(1,1) = -y*z / (d2 * dxy);
    J(1,2) = dxy / d2;

    J(2,0) = x / d;
    J(2,1) = y / d;
    J(2,2) = z / d;
    return J;
}