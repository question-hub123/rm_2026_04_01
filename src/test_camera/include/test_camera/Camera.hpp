#pragma once
#include "CameraApi.h"
#include "opencv2/core/core.hpp"
#include <string>
 
class CameraWrapper {
private:
    int hCamera;                  // 相机句柄
    unsigned char* g_pRgbBuffer;  // 处理后数据缓存区
    tSdkCameraCapbility tCapability;  // 设备描述信息
    int channel;                  // 图像通道数
    bool isInitialized;           // 相机是否初始化成功
 
public:
    // 构造函数和析构函数
    CameraWrapper();
    ~CameraWrapper();
 
    // 初始化相机
    bool init(int index = 0);
 
    // 关闭相机
    void close();
 
    // 获取一帧图像（返回OpenCV格式）
    bool getFrame(cv::Mat& frame, int timeout = 1000);
 
    // 检查相机是否已初始化
    bool isOpen() const { return isInitialized; }

    //设置曝光时间
    bool setExposure(float exposureMs);
};