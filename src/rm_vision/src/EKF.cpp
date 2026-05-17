#include "EKF.hpp"

EKF::EKF() : initialized_(false) {
    x.setZero();
    P = Eigen::Matrix<double, 10, 10>::Identity() * 1.0;
    F = Eigen::Matrix<double, 10, 10>::Identity();
    H.setZero();

    // 过程噪声协方差 Q
    Q.setIdentity();
    // xc, vxc, yc, vyc, zc, vzc, yaw, vyaw, r, zp
    Q.diagonal() << 0.01, 0.001,  // x
                    0.01, 0.001,  // y
                    0.01, 0.001,  // z
                    0.02, 0.2,  // yaw
                    1e-6,      // r  (半径变化极小)
                    1e-6;      // zp (装甲板高度偏移变化极小)

    // 测量噪声协方差 R (极其重要)
    // 观测维度: [yaw_cam, pitch_cam, distance, orientation_yaw]
    R.setIdentity();
    // 策略：极度信任位置系观测(yaw_cam, pitch_cam, dist)，极度不信任PnP的单帧姿态(orientation_yaw)
    R.diagonal() << 0.001, 0.001, 4.0, 0.005; 
}

void EKF::init(const Eigen::Vector3d& p_armor, double armor_yaw, double r0, double zp0) {
    x.setZero();
    // 根据装甲板位置和初始半径推算车体中心
    x(0) = p_armor.x() - r0 * cos(armor_yaw); // xc
    x(2) = p_armor.y() - r0 * sin(armor_yaw); // yc
    x(4) = p_armor.z() - zp0;                 // zc
    
    x(6) = armor_yaw; // yaw
    x(8) = r0;        // r
    x(9) = zp0;       // zp

    initialized_ = true;
}

void EKF::predict(double dt) {
    if (!initialized_ || dt <= 0.0) return;

    // 状态转移: 匀速运动模型
    x(0) += x(1) * dt; // xc += vxc*dt
    x(2) += x(3) * dt; // yc += vyc*dt
    x(4) += x(5) * dt; // zc += vzc*dt
    x(6) += x(7) * dt; // yaw += vyaw*dt (连续变化)

    // 半径 r 和高度偏移 zp 保持不变

    // 更新雅可比矩阵 F
    F.setIdentity();
    F(0, 1) = dt;
    F(2, 3) = dt;
    F(4, 5) = dt;
    F(6, 7) = dt;

    P = F * P * F.transpose() + Q;
}

void EKF::update(const Eigen::Vector4d& z) 
{
    if (!initialized_) return;

    // 预测观测 h(x) -> [yaw_cam, pitch_cam, dist, orientation_yaw]
    auto h_func = [](const Eigen::VectorXd& x_val) -> Eigen::Vector4d {
        double xc_v  = x_val(0);
        double yc_v  = x_val(2);
        double zc_v  = x_val(4);
        double yv    = x_val(6); // 正在追踪的装甲板的连续yaw
        double rr    = x_val(8); // 半径
        double zp_v  = x_val(9); // 高度偏移

        // 测算装甲板坐标 (世界系下，但因为最终转球面，相对原点)
        double xa_v = xc_v + rr * cos(yv);
        double ya_v = yc_v + rr * sin(yv);
        double za_v = zc_v + zp_v;

        double dxy = sqrt(xa_v * xa_v + ya_v * ya_v);
        double dd  = sqrt(xa_v * xa_v + ya_v * ya_v + za_v * za_v);
        
        Eigen::Vector4d zz;
        zz(0) = atan2(ya_v, xa_v); // 球面坐标系偏航角
        zz(1) = atan2(za_v, dxy);  // 球面坐标系俯仰角
        zz(2) = dd;                // 距离
        zz(3) = yv;                // 装甲板自身朝向
        return zz;
    };

    Eigen::Vector4d z_pred = h_func(x);
    H = numericalJacobian(x, h_func);

    // 卡方检验
    Eigen::Matrix4d S = H * P * H.transpose() + R;
    Eigen::Vector4d y_res = z - z_pred;
    
    // 角度残差必须归一化到 [-pi, pi]
    y_res(0) = normalizeAngle(y_res(0));
    y_res(1) = normalizeAngle(y_res(1));
    y_res(3) = normalizeAngle(y_res(3));

    double mahalanobis = y_res.transpose() * S.inverse() * y_res;
    const double chi2_threshold = 80; // 4 自由度，99% 置信度
    if (mahalanobis > chi2_threshold) {
        // 残差过大，可能是误检或跳变，放弃本次更新
        return;
    }

    // 卡尔曼增益
    Eigen::Matrix<double, 10, 4> K = P * H.transpose() * S.inverse();
    x += K * y_res;
    P = (Eigen::Matrix<double, 10, 10>::Identity() - K * H) * P;

    // 参数钳制保护
    if (x(8) < 0.12) x(8) = 0.12;
    if (x(8) > 0.45) x(8) = 0.45;
    if (x(9) < -0.2) x(9) = -0.2;
    if (x(9) > 0.2)  x(9) =  0.2;
}

Eigen::Vector3d EKF::getArmorPosition() const {
    double yaw = x(6);
    double r = x(8);
    return Eigen::Vector3d(x(0) + r * cos(yaw), x(2) + r * sin(yaw), x(4) + x(9));
}

Eigen::Vector3d EKF::getVehiclePosition() const { return Eigen::Vector3d(x(0), x(2), x(4)); }
double EKF::getArmorZOffset() const { return x(9); }
double EKF::getRadius() const { return x(8); }
double EKF::getContinuousYaw() const { return x(6); }

double EKF::normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

double EKF::shortestAngularDistance(double from, double to) {
    return normalizeAngle(to - from);
}

Eigen::MatrixXd EKF::numericalJacobian(
    const Eigen::VectorXd& x0,
    const std::function<Eigen::Vector4d(const Eigen::VectorXd&)>& h_func,
    double eps) const 
{
    int n = x0.size();       
    int m = 4;
    Eigen::MatrixXd J(m, n);
    Eigen::VectorXd x_pert = x0;
    for (int i = 0; i < n; ++i) {
        x_pert(i) += eps;
        Eigen::Vector4d h_plus = h_func(x_pert);
        x_pert(i) -= 2.0 * eps;
        Eigen::Vector4d h_minus = h_func(x_pert);
        x_pert(i) += eps; 
        
        // 注意处理角度的跃变
        Eigen::Vector4d diff = h_plus - h_minus;
        diff(0) = normalizeAngle(diff(0));
        diff(1) = normalizeAngle(diff(1));
        diff(3) = normalizeAngle(diff(3));
        
        J.col(i) = diff / (2.0 * eps);
    }
    return J;
}