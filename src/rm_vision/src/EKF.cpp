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
    R.diagonal() << 0.01, 0.01, 0.2, 0.1;
}

void EKF::init(const Eigen::Vector3d& p_armor, double armor_yaw, double r0) {
    x.setZero();
    x(0) = p_armor.x() - r0 * cos(armor_yaw);
    x(2) = p_armor.y() - r0 * sin(armor_yaw);
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

void EKF::update(const Eigen::Vector4d& z)
{
    if (!initialized_) return;

    double yaw_vehicle = x(6);
    double r = x(8);
    double xc = x(0);
    double yc = x(2);
    double zc = x(4);

    // ===== 预测观测 h(x)：从状态算装甲板球坐标 =====
    double xa = xc + r * cos(yaw_vehicle);
    double ya = yc + r * sin(yaw_vehicle);
    double za = zc;

    double dist_xy = sqrt(xa * xa + ya * ya);
    double dist = sqrt(xa * xa + ya * ya + za * za);
    double pred_yaw   = atan2(ya, xa);
    double pred_pitch = atan2(za, dist_xy);
    // 装甲板法线朝向（车头指向）即为车体 yaw
    double pred_armor_yaw = yaw_vehicle;

    Eigen::Vector4d z_pred(pred_yaw, pred_pitch, dist, pred_armor_yaw);

    // ===== 数值雅可比 H (4x9) =====
    auto h_func = [this](const Eigen::VectorXd& x_val) -> Eigen::Vector4d {
        double yv = x_val(6);
        double rr = x_val(8);
        double xa_v = x_val(0) + rr * cos(yv);
        double ya_v = x_val(2) + rr * sin(yv);
        double za_v = x_val(4);
        double dxy = sqrt(xa_v * xa_v + ya_v * ya_v);
        double dd  = sqrt(xa_v * xa_v + ya_v * ya_v + za_v * za_v);
        Eigen::Vector4d zz;
        zz(0) = atan2(ya_v, xa_v);
        zz(1) = atan2(za_v, dxy);
        zz(2) = dd;
        zz(3) = yv;
        return zz;
    };
    H = numericalJacobian(x, h_func);

    // ===== 卡方检验（拒绝野值） =====
    Eigen::Matrix4d S = H * P * H.transpose() + R;
    Eigen::Vector4d y_res = z - z_pred;
    // 角度残差归一化到[-pi, pi]
    y_res(0) = normalizeAngle(y_res(0));
    y_res(3) = normalizeAngle(y_res(3));
    double mahalanobis = y_res.transpose() * S.inverse() * y_res;
    const double chi2_threshold = 9.488; // 4 自由度，95% 置信度
    if (mahalanobis > chi2_threshold) {
        // 观测异常，跳过更新（仅靠 predict 传播状态）
        return;
    }

    // ===== 卡尔曼更新 =====
    Eigen::Matrix<double, 9, 4> K = P * H.transpose() * S.inverse();
    x += K * y_res;
    // 角度归一化
    x(6) = normalizeAngle(x(6));
    P = (Eigen::Matrix<double, 9, 9>::Identity() - K * H) * P;

    // 半径钳制
    if (x(8) < 0.12) x(8) = 0.12;
    if (x(8) > 0.4)  x(8) = 0.4;
}

Eigen::Vector3d EKF::getArmorPosition() const {
    double yaw = x(6);
    double r = x(8);
    return Eigen::Vector3d(x(0) + r * cos(yaw), x(2) + r * sin(yaw), x(4));
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

double EKF::shortestAngularDistance(double from, double to) 
{
    return normalizeAngle(to - from);
}

Eigen::MatrixXd EKF::numericalJacobian(
        const Eigen::VectorXd& x0,
        const std::function<Eigen::Vector4d(const Eigen::VectorXd&)>& h_func,
        double eps) const
    {
        int n = x0.size();       // 9
        int m = 4;
        Eigen::MatrixXd J(m, n);
        Eigen::VectorXd x_pert = x0;
        for (int i = 0; i < n; ++i) {
            x_pert(i) += eps;
            Eigen::Vector4d h_plus = h_func(x_pert);
            x_pert(i) -= 2.0 * eps;
            Eigen::Vector4d h_minus = h_func(x_pert);
            x_pert(i) += eps; // 恢复
            J.col(i) = (h_plus - h_minus) / (2.0 * eps);
        }
        return J;
    }