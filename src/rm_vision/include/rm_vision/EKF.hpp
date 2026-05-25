// ekf11.hpp
#pragma once
#include <eigen3/Eigen/Dense>
#include <functional>

/**
 * @brief 与同济 sp_vision_25 完全一致的 11 维扩展卡尔曼滤波器
 *
 * 状态向量 x (11): [xc, vxc, yc, vyc, zc, vzc, yaw, vyaw, r, l, h]
 *   - xc, yc, zc : 装甲板旋转中心在世界坐标系下的位置
 *   - vxc, vyc, vzc : 旋转中心的速度
 *   - yaw : 当前旋转角度（对应第 0 号装甲板的朝向，连续化）
 *   - vyaw : 自转角速度
 *   - r  : 旋转半径（基础短半径）
 *   - l  : 长半径增量（对四装甲板，id=1,3 时半径为 r+l）
 *   - h  : 高度差（高板与低板的 z 差）
 */
class EKF11 {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix<double, 11, 1> x;    // 状态
    Eigen::Matrix<double, 11, 11> P;   // 协方差

    EKF11();

    /**
     * @brief 用首次检测初始化滤波器
     * @param p_armor_world  装甲板中心在世界系下的坐标
     * @param armor_roll     装甲板自身的旋转角 (ypr[0])
     * @param armor_num      装甲板数量 (通常 4)
     * @param init_r         初始短半径
     * @param init_l         初始长半径增量
     * @param init_h         初始高度差
     * @param P0_diag        初始协方差对角元素 (11 维)
     */
    void init(const Eigen::Vector3d& p_armor_world,
              double armor_roll,
              int armor_num,
              double init_r = 0.26,
              double init_l = 0.0,
              double init_h = 0.0,
              const Eigen::Matrix<double, 11, 1>& P0_diag = Eigen::Matrix<double, 11, 1>::Constant(0.1));

    /**
     * @brief 预测一步
     * @param dt           时间间隔 (秒)
     * @param is_outpost   是否为前哨站（使用不同噪声参数）
     */
    void predict(double dt);

    /**
     * @brief 更新步骤（包含装甲板 id 匹配）
     * @param z_obs        观测向量 [ypd_yaw, ypd_pitch, distance, armor_roll]
     *                      其中 ypd 为装甲板中心在世界系下的球坐标表示
     * @param armor_xyz    装甲板中心在世界系下的笛卡尔坐标（用于匹配和动态噪声）
     */
    void update(const Eigen::Vector4d& z_obs, const Eigen::Vector3d& armor_xyz);

    // ---- 工具函数 ----
    Eigen::Vector3d getArmorCenter(int id) const;         // 获取第 id 块装甲板中心世界坐标
    Eigen::Vector3d getArmorCenter() const;               // 获取当前跟踪的装甲板中心（使用 last_id）
    double getYaw() const { return x(6); }
    double getVyaw() const { return x(7); }
    double getR() const { return x(8); }
    double getL() const { return x(9); }
    double getH() const { return x(10); }

    static double limitRad(double angle);
    static Eigen::Vector3d xyz2ypd(const Eigen::Vector3d& xyz);
    static Eigen::Matrix<double, 3, 3> xyz2ypdJacobian(const Eigen::Vector3d& xyz);

    Eigen::Vector3d predictFutureCenter(double dt) const;
    Eigen::Vector3d predictFutureArmor(int id, double dt) const;

private:
    int armor_num_;      // 装甲板数量
    int last_id_;        // 上一帧匹配到的 id
    int update_count_;   // 更新次数
    bool converged_;     // 是否已收敛

    // 计算第 id 块装甲板中心的世界坐标
    Eigen::Vector3d hArmorXyz(const Eigen::Matrix<double, 11, 1>& state, int id) const;

    // 观测函数 h(x) → 4 维
    Eigen::Vector4d h_func(const Eigen::Matrix<double, 11, 1>& state, int id) const;

    // 解析观测雅可比 dh/dx (4x11)
    Eigen::Matrix<double, 4, 11> h_jacobian(const Eigen::Matrix<double, 11, 1>& state, int id) const;
};