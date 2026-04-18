#pragma once
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>

class KF
{
public:
    KF(double dt = 0.33);
    
    void init(double yaw, double pitch);//初始化状态和协方差

    void predict(double dt);

    void update(double yaw_meas, double pitch_meas);

    void getState(double& yaw, double& yaw_rate, double& pitch, double& pitch_rate)const;

    bool isInititalized() const{ return initialized_;}


private:
    bool initialized_;
    double dt_;

    Eigen::VectorXd X_; // 状态向量 [yaw, yaw_rate, pitch, pitch_rate]ᵀ

    Eigen::MatrixXd P_;// 状态协方差矩阵

    Eigen::MatrixXd A_;// 状态转移矩阵 A

    Eigen::Matrix<double, 2, 4> H_;// 观测矩阵 H
    
    Eigen::MatrixXd Q_;// 过程噪声协方差 Q
    
    Eigen::MatrixXd R_;// 测量噪声协方差 R

    double normalizeAngle(double angle);
};