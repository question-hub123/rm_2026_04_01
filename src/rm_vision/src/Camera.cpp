#include "Camera.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#define CHECK_GX(status, msg) \
    if (status != GX_STATUS_SUCCESS) { \
        printf("[ERROR] " msg " failed! GX_STATUS: %d\n", status); \
        return false; \
    }

CameraWrapper::CameraWrapper() 
    : hCamera(nullptr), g_pRgbBuffer(nullptr), 
      width(0), height(0), channel(3), 
      isInitialized(false), isStreaming(false) {
}

CameraWrapper::~CameraWrapper() {
    close();
}

bool CameraWrapper::init(int index) {
    if (isInitialized) close();

    GX_STATUS status = GX_STATUS_SUCCESS;
    uint32_t iCameraCounts = 0;

    // 1. 初始化SDK
    status = GXInitLib();
    CHECK_GX(status, "GXInitLib");

    // 2. 枚举设备
    status = GXUpdateDeviceList(&iCameraCounts, 1000);
    CHECK_GX(status, "GXUpdateDeviceList");

    if (iCameraCounts <= 0) {
        printf("[ERROR] No camera found!\n");
        GXCloseLib();
        return false;
    }

    // 索引转换（迈德0 -> 大恒1）
    int dah_index = index + 1;
    if (dah_index < 1 || dah_index > (int)iCameraCounts) {
        printf("[ERROR] Invalid index: %d\n", index);
        GXCloseLib();
        return false;
    }

    // 3. 打开设备
    status = GXOpenDeviceByIndex(dah_index, &hCamera);
    CHECK_GX(status, "GXOpenDeviceByIndex");

    // 4. 获取分辨率
    int64_t w = 0, h = 0;
    GXGetInt(hCamera, GX_INT_WIDTH, &w);
    GXGetInt(hCamera, GX_INT_HEIGHT, &h);
    width = static_cast<int>(w);
    height = static_cast<int>(h);

    // 5. 设置采集模式和触发模式
    GXSetEnum(hCamera, GX_ENUM_ACQUISITION_MODE, GX_ACQ_MODE_CONTINUOUS);
    GXSetEnum(hCamera, GX_ENUM_TRIGGER_MODE, GX_TRIGGER_MODE_OFF);

    // 6. 设置像素格式
    GXSetEnum(hCamera, GX_ENUM_PIXEL_FORMAT, GX_PIXEL_FORMAT_BAYER_RG8);
    channel = 3;

    // 7. 分配RGB缓冲区
    g_pRgbBuffer = (unsigned char*)malloc(width * height * 3);
    if (g_pRgbBuffer == nullptr) {
        printf("[ERROR] Malloc failed!\n");
        GXCloseDevice(hCamera);
        GXCloseLib();
        return false;
    }

    // 8. 设置采集缓冲数量
    GXSetAcqusitionBufferNumber(hCamera, 10);

    // 9. 开始采集
    status = GXSendCommand(hCamera, GX_COMMAND_ACQUISITION_START);
    CHECK_GX(status, "Start acquisition");
    isStreaming = true;

    isInitialized = true;
    printf("[INFO] Camera init ok! %dx%d\n", width, height);
    return true;
}

void CameraWrapper::close() {
    if (isInitialized) {
        if (isStreaming) {
            GXSendCommand(hCamera, GX_COMMAND_ACQUISITION_STOP);
            isStreaming = false;
        }
        if (hCamera) {
            GXCloseDevice(hCamera);
            hCamera = nullptr;
        }
        GXCloseLib();
        isInitialized = false;
    }
    if (g_pRgbBuffer) {
        free(g_pRgbBuffer);
        g_pRgbBuffer = nullptr;
    }
}

bool CameraWrapper::getFrame(cv::Mat& frame, int timeout) {
    // 1. 检查相机状态
    if (!isInitialized || !isStreaming) {
        printf("[ERROR] Camera not ready!\n");
        return false;
    }

    // 2. 定义并初始化帧数据结构体
    GX_FRAME_DATA frameData;
    memset(&frameData, 0, sizeof(GX_FRAME_DATA));

    // 3. 获取一帧图像
    GX_STATUS status = GXGetImage(hCamera, &frameData, timeout);
    if (status != GX_STATUS_SUCCESS) {
        // 只有非超时错误才打印
        if (status != GX_STATUS_TIMEOUT) {
            printf("[ERROR] GXGetImage failed! Error code: %d\n", status);
        }
        return false;
    }

    // 4. 检查帧状态是否正常
    if (frameData.nStatus != GX_FRAME_STATUS_SUCCESS) {
        printf("[ERROR] Frame status invalid! Status: %d\n", frameData.nStatus);
        return false;
    }

    // 5. 提取图像数据指针和尺寸（强转为SDK要求的类型）
    void* pRawBuf = frameData.pImgBuf;
    VxUint32 imgW = static_cast<VxUint32>(frameData.nWidth);
    VxUint32 imgH = static_cast<VxUint32>(frameData.nHeight);

    // 6. 显式声明枚举变量（确保类型匹配）
    DX_BAYER_CONVERT_TYPE convertType = RAW2RGB_NEIGHBOUR;
    DX_PIXEL_COLOR_FILTER bayerType = BAYERRG;

    // 7. 调用Bayer转RGB函数
    VxInt32 dxStatus = DxRaw8toRGB24(
        pRawBuf,
        g_pRgbBuffer,
        imgW,
        imgH,
        convertType,
        bayerType,
        false
    );

    if (dxStatus != DX_OK) {
        printf("[ERROR] DxRaw8toRGB24 failed! Error code: %d\n", dxStatus);
        return false;
    }

    // 8. 将RGB数据转换为OpenCV的BGR格式
    cv::Mat rgbMat(
        static_cast<int>(imgH), 
        static_cast<int>(imgW), 
        CV_8UC3, 
        g_pRgbBuffer
    );
    cv::cvtColor(rgbMat, frame, cv::COLOR_RGB2BGR);

    // 9. 返回成功
    return !frame.empty();
}



bool CameraWrapper::setExposure(float exposureMs) {
    if (!isInitialized) return false;

    // 关闭自动曝光
    GXSetEnum(hCamera, GX_ENUM_EXPOSURE_AUTO, GX_EXPOSURE_AUTO_OFF);

    // 设置曝光时间 (ms -> us)
    double exposureUs = exposureMs * 1000.0;
    GX_STATUS status = GXSetFloat(hCamera, GX_FLOAT_EXPOSURE_TIME, exposureUs);
    
    if (status == GX_STATUS_SUCCESS) {
        printf("[INFO] Exposure set to %.1f ms\n", exposureMs);
        return true;
    }
    return false;
}
