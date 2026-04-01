#pragma once
#include "GxIAPI.h"
#include "DxImageProc.h"
#include "opencv2/core/core.hpp"
#include <string>
 
class CameraWrapper {
private:
    GX_DEV_HANDLE hCamera;                  // 相机句柄
    unsigned char* g_pRgbBuffer;            // RGB缓存区
    int width;                               // 图像宽度
    int height;                              // 图像高度
    int channel;                             // 图像通道数
    bool isInitialized;                      // 是否初始化
    bool isStreaming;                        // 是否正在采集
 
public:
    CameraWrapper();
    ~CameraWrapper();
 
    bool init(int index = 0);
    void close();
    bool getFrame(cv::Mat& frame, int timeout = 1000);
    bool isOpen() const { return isInitialized; }
    bool setExposure(float exposureMs);
};
