
#include <eigen3/Eigen/Dense>
#include <cmath>

class EKF {
public:
    // 状态向量: [xc, vxc, yc, vyc, za, vza, yaw, v_yaw, r]^T
    // xc, yc: 车体中心在世界坐标系下的X, Y坐标
    // za: 装甲板的高度 (Z坐标)
    // yaw: 车体朝向角 (连续化, 非 -pi~pi)
    // r: 旋转半径 (装甲板到车体中心的距离)
    Eigen::Matrix<double, 9, 1> x;
    Eigen::Matrix<double, 9, 9> P;
    Eigen::Matrix<double, 9, 9> F;
    Eigen::Matrix<double, 4, 9> H;
    Eigen::Matrix<double, 9, 9> Q;
    Eigen::Matrix<double, 4, 4> R;

    bool initialized_;

    EKF();

    void init(const Eigen::Vector3d& p_armor, double armor_yaw, double r0 = 0.26);

    void predict(double dt);

    void update(const Eigen::Vector4d& z);

    // 从当前状态反推装甲板在世界系下的坐标
    Eigen::Vector3d getArmorPosition() const;

    // 获取滤波后的车体中心位置
    Eigen::Vector3d getVehiclePosition() const;

    // 获取当前估计的半径
    double getRadius() const;

    // 获取连续化后的车体yaw (可能超出-pi~pi, 但代表累计转角)
    double getContinuousYaw() const;

    static double normalizeAngle(double angle);
    
    // 用于连续化观测yaw的辅助函数
    static double shortestAngularDistance(double from, double to);

    Eigen::MatrixXd numericalJacobian(const Eigen::VectorXd& x0,
        const std::function<Eigen::Vector4d(const Eigen::VectorXd&)>& h_func,
        double eps = 1e-6) const;
};