// Camera.hpp
#pragma once
#include "GxIAPI.h"
#include "DxImageProc.h"
#include "opencv2/core/core.hpp"
#include <string>

class CameraWrapper {
private:
    GX_DEV_HANDLE hCamera;          // 设备句柄
    GX_DS_HANDLE hStream;           // 数据流句柄
    unsigned char* g_pRgbBuffer;    // RGB缓存
    int width, height, channel;
    uint32_t payloadSize;
    bool isInitialized, isStreaming;
    int64_t colorFilter;            // Bayer格式（从相机读取）

public:
    CameraWrapper();
    ~CameraWrapper();
    bool init(int index = 0);
    void close();
    bool getFrame(cv::Mat& frame, int timeout = 1000);
    bool setExposure(float exposureMs);
    bool isOpen() const { return isInitialized; }
};