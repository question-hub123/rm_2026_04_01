#pragma once
#include <eigen3/Eigen/Dense>
#include <cmath>

class EKF {
public:
    EKF();

    // 初始化：传入观测到的装甲板世界坐标 (m) 和世界系装甲板朝向 (rad)
    void init(const Eigen::Vector3d& p_armor, double yaw_abs);

    // 预测：匀加速模型，dt 为时间间隔 (s)
    void predict(double dt);

    // 更新：观测装甲板的世界坐标和世界系朝向
    void update(const Eigen::Vector3d& p_armor, double yaw_abs);

    // 获取滤波后的车体状态（位置、偏航、角速度、半径）
    void getState(Eigen::Vector3d& pos_c, double& yaw, double& v_yaw, double& r) const;

    // 访问接口
    double getVx()    const { return X_(IDX_VX); }
    double getVy()    const { return X_(IDX_VY); }
    double getVz()    const { return X_(IDX_VZ); }
    double getAx()    const { return X_(IDX_AX); }
    double getAy()    const { return X_(IDX_AY); }
    double getAz()    const { return X_(IDX_AZ); }
    double getAYaw()  const { return X_(IDX_AYAW); }
    double getR()     const { return X_(IDX_R); }

    bool isInitialized() const { return initialized_; }

private:
    bool initialized_;

    // 状态索引 (13 维)
    static constexpr int IDX_X    = 0;
    static constexpr int IDX_Y    = 1;
    static constexpr int IDX_Z    = 2;
    static constexpr int IDX_VX   = 3;
    static constexpr int IDX_VY   = 4;
    static constexpr int IDX_VZ   = 5;
    static constexpr int IDX_AX   = 6;
    static constexpr int IDX_AY   = 7;
    static constexpr int IDX_AZ   = 8;
    static constexpr int IDX_YAW  = 9;
    static constexpr int IDX_VYAW = 10;
    static constexpr int IDX_AYAW = 11;
    static constexpr int IDX_R    = 12;   // 车体半径

    Eigen::VectorXd X_;                     // 13 维状态
    Eigen::MatrixXd P_;                     // 13x13 协方差
    Eigen::MatrixXd Q_;                     // 13x13 过程噪声
    Eigen::MatrixXd R_;                     // 4x4  测量噪声

    double normalizeAngle(double angle) const;
};