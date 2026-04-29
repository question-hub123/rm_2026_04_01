// Camera.cpp
#include "Camera.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <chrono>

#define GX_VERIFY(status, msg) \
    if (status != GX_STATUS_SUCCESS) { \
        printf("[ERROR] " msg " failed! GX_STATUS: %d\n", status); \
        return false; \
    }

CameraWrapper::CameraWrapper() 
    : hCamera(nullptr), hStream(nullptr), g_pRgbBuffer(nullptr),
      width(0), height(0), channel(3), payloadSize(0),
      isInitialized(false), isStreaming(false), colorFilter(GX_COLOR_FILTER_NONE) {}

CameraWrapper::~CameraWrapper() { close(); }

bool CameraWrapper::init(int index) {
    if (isInitialized) close();

    GX_STATUS status = GXInitLib();
    GX_VERIFY(status, "GXInitLib");

    uint32_t devCount = 0;
    status = GXUpdateAllDeviceList(&devCount, 1000);
    GX_VERIFY(status, "GXUpdateAllDeviceList");
    if (devCount == 0) {
        printf("[ERROR] No camera found!\n");
        GXCloseLib();
        return false;
    }

    int openIdx = index + 1;
    status = GXOpenDeviceByIndex(openIdx, &hCamera);
    GX_VERIFY(status, "GXOpenDeviceByIndex");

    // 获取分辨率
    int64_t w = 0, h = 0;
    GXGetInt(hCamera, GX_INT_WIDTH, &w);
    GXGetInt(hCamera, GX_INT_HEIGHT, &h);
    width = static_cast<int>(w);
    height = static_cast<int>(h);

    // 获取 Bayer 格式（从相机读取，不要硬编码）
    GX_ENUM_VALUE enumVal;
    status = GXGetEnumValue(hCamera, "PixelColorFilter", &enumVal);
    if (status == GX_STATUS_SUCCESS) {
        colorFilter = enumVal.stCurValue.nCurValue;
    } else {
        colorFilter = GX_COLOR_FILTER_BAYER_RG; // 默认值
    }

    // 获取流通道句柄
    uint32_t dsNum = 0;
    status = GXGetDataStreamNumFromDev(hCamera, &dsNum);
    GX_VERIFY(status, "GXGetDataStreamNumFromDev");
    if (dsNum < 1) {
        printf("[ERROR] No data stream found!\n");
        GXCloseDevice(hCamera);
        GXCloseLib();
        return false;
    }
    status = GXGetDataStreamHandleFromDev(hCamera, 1, &hStream);
    GX_VERIFY(status, "GXGetDataStreamHandleFromDev");

    // 获取 PayloadSize
    status = GXGetPayLoadSize(hStream, &payloadSize);
    GX_VERIFY(status, "GXGetPayLoadSize");
    printf("[INFO] PayloadSize = %u\n", payloadSize);

    // 设置采集模式与触发
    GXSetEnumValueByString(hCamera, "AcquisitionMode", "Continuous");
    GXSetEnumValueByString(hCamera, "TriggerMode", "Off");

    // 设置流参数（可选但推荐）
    GXSetIntValue(hCamera, "StreamTransferSize", 64 * 1024);
    GXSetIntValue(hCamera, "StreamTransferNumberUrb", 64);

    // 设置 Buffer 数量
    GXSetAcqusitionBufferNumber(hCamera, 2);                     // 最小缓冲
    GXSetEnumValueByString(hCamera, "AcquisitionFrameRateMode", "Off"); // 让相机自由运行

    // 分配 RGB 缓冲区
    g_pRgbBuffer = new unsigned char[payloadSize * 3];
    if (!g_pRgbBuffer) {
        printf("[ERROR] Malloc failed!\n");
        GXCloseDevice(hCamera);
        GXCloseLib();
        return false;
    }

    // 启动流采集
    status = GXStreamOn(hCamera);
    GX_VERIFY(status, "GXStreamOn");

    // 等待相机稳定出图
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    isStreaming = true;
    isInitialized = true;
    printf("[INFO] Camera initialized successfully. %dx%d, BayerType=%lld\n", width, height, colorFilter);
    return true;
}

void CameraWrapper::close() {
    if (isStreaming) {
        GXStreamOff(hCamera);
        isStreaming = false;
    }
    if (hCamera) {
        GXCloseDevice(hCamera);
        hCamera = nullptr;
    }
    GXCloseLib();
    isInitialized = false;
    if (g_pRgbBuffer) {
        delete[] g_pRgbBuffer;
        g_pRgbBuffer = nullptr;
    }
}

bool CameraWrapper::getFrame(cv::Mat& frame, int timeout) {
    if (!isInitialized || !isStreaming) return false;

    // ✅ 改动1：取图前清空缓冲队列，丢弃旧帧，保证实时性
    GXFlushQueue(hCamera);

    PGX_FRAME_BUFFER pFrameBuf = nullptr;
    GX_STATUS status = GXDQBuf(hCamera, &pFrameBuf, timeout);
    if (status != GX_STATUS_SUCCESS) {
        if (status != GX_STATUS_TIMEOUT)
            printf("[ERROR] GXDQBuf failed! Error code: %d\n", status);
        return false;
    }

    if (pFrameBuf->nStatus != GX_FRAME_STATUS_SUCCESS) {
        GXQBuf(hCamera, pFrameBuf);
        return false;
    }

    VxInt32 ret = DX_OK;
    if (pFrameBuf->nPixelFormat == GX_PIXEL_FORMAT_BAYER_GR8 ||
        pFrameBuf->nPixelFormat == GX_PIXEL_FORMAT_BAYER_RG8 ||
        pFrameBuf->nPixelFormat == GX_PIXEL_FORMAT_BAYER_GB8 ||
        pFrameBuf->nPixelFormat == GX_PIXEL_FORMAT_BAYER_BG8) {
        
        // ✅ 改动2：使用 DxRaw8toRGB24Ex 直接输出 BGR，避免后续 cvtColor
        ret = DxRaw8toRGB24Ex(
            (unsigned char*)pFrameBuf->pImgBuf,
            g_pRgbBuffer,
            pFrameBuf->nWidth, pFrameBuf->nHeight,
            RAW2RGB_NEIGHBOUR,
            DX_PIXEL_COLOR_FILTER(colorFilter),
            false,
            DX_ORDER_BGR);          // 关键参数：直接输出 BGR 顺序
    } else {
        printf("[ERROR] Unsupported pixel format: %d\n", pFrameBuf->nPixelFormat);
        GXQBuf(hCamera, pFrameBuf);
        return false;
    }

    if (ret != DX_OK) {
        printf("[ERROR] DxRaw8toRGB24Ex failed! ret=%d\n", ret);
        GXQBuf(hCamera, pFrameBuf);
        return false;
    }

    // ✅ 改动3：直接使用转换好的 BGR 数据构造 cv::Mat，无需再转换
    // 注意：这里需要深拷贝一份数据，因为 g_pRgbBuffer 在下一次取图时会被覆盖
    frame = cv::Mat(pFrameBuf->nHeight, pFrameBuf->nWidth, CV_8UC3, g_pRgbBuffer).clone();

    GXQBuf(hCamera, pFrameBuf);
    return true;
}

bool CameraWrapper::setExposure(float exposureMs) {
    if (!isInitialized) return false;
    GXSetEnumValueByString(hCamera, "ExposureAuto", "Off");
    double exposureUs = exposureMs * 1000.0;
    GX_STATUS status = GXSetFloatValue(hCamera, "ExposureTime", exposureUs);
    if (status == GX_STATUS_SUCCESS) {
        printf("[INFO] Exposure set to %.1f ms\n", exposureMs);
        return true;
    }
    return false;
}

bool CameraWrapper::setGain(float gainDb) {
    if (!isInitialized) return false;
    // 关闭自动增益
    GXSetEnumValueByString(hCamera, "GainAuto", "Off");
    GX_STATUS status = GXSetFloatValue(hCamera, "Gain", gainDb);
    if (status == GX_STATUS_SUCCESS) {
        printf("[INFO] Gain set to %.2f dB\n", gainDb);
        return true;
    } else {
        printf("[ERROR] Set Gain failed! status = %d\n", status);
        return false;
    }
}