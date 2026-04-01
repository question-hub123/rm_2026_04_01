#pragma once
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>

class KF
{
public:
    KF(double dt = 0.033);
    
    void init(double x, double y, double yaw);//初始化状态和协方差

    void predict(double dt);

    void update(double zx, double zy, double zYaw);

    void getState(double& x, double& y, double& vx, double& vy, double& yaw)const;

    bool isInititalized() const{ return initialized_;}


private:
    bool initialized_ = false;
    double dt_;

    Eigen::VectorXd X_; // 状态向量 [x, y, vx, vy, yaw]ᵀ

    Eigen::MatrixXd P_;// 状态协方差矩阵

    Eigen::MatrixXd A_;// 状态转移矩阵 A

    Eigen::MatrixXd H_;// 观测矩阵 H
    
    Eigen::MatrixXd Q_;// 过程噪声协方差 Q
    
    Eigen::MatrixXd R_;// 测量噪声协方差 R
};