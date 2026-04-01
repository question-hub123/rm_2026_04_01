#include "Camera.hpp"
#include "CameraApi.h"
#include "CameraStatus.h"
#include <stdio.h>
#include <stdlib.h>
 
CameraWrapper::CameraWrapper() 
    : hCamera(-1), g_pRgbBuffer(nullptr), channel(3), isInitialized(false) {
}
 
CameraWrapper::~CameraWrapper() {
    close();
}
 
bool CameraWrapper::init(int index) {
    // 如果已经初始化，则先关闭
    if (isInitialized) {
        close();
    }
 
    int iStatus = -1;
    tSdkCameraDevInfo tCameraEnumList;
    int iCameraCounts = 1;
 
    // 初始化SDK
    CameraSdkInit(1);
 
    // 枚举设备
    iStatus = CameraEnumerateDevice(&tCameraEnumList, &iCameraCounts);
    if (iStatus != CAMERA_STATUS_SUCCESS) {
        printf("CameraEnumerateDevice failed, status: %d\n", iStatus);
        return false;
    }
 
    // 检查设备数量
    if (iCameraCounts <= 0) {
        printf("No camera found!\n");
        return false;
    }
 
    // 检查索引是否有效
    if (index < 0 || index >= iCameraCounts) {
        printf("Invalid camera index: %d\n", index);
        return false;
    }
 
    // 初始化相机
    iStatus = CameraInit(&tCameraEnumList, -1, -1, &hCamera);
    if (iStatus != CAMERA_STATUS_SUCCESS) {
        printf("CameraInit failed, status: %d\n", iStatus);
        return false;
    }
 
    // 获取相机特性
    CameraGetCapability(hCamera, &tCapability);
 
    // 分配RGB缓冲区
    g_pRgbBuffer = (unsigned char*)malloc(
        tCapability.sResolutionRange.iHeightMax * 
        tCapability.sResolutionRange.iWidthMax * 3
    );
    if (g_pRgbBuffer == nullptr) {
        printf("Failed to allocate memory for image buffer\n");
        CameraUnInit(hCamera);
        return false;
    }
 
    // 开始播放
    CameraPlay(hCamera);
 
    // 设置输出格式
    if (tCapability.sIspCapacity.bMonoSensor) {
        channel = 1;
        CameraSetIspOutFormat(hCamera, CAMERA_MEDIA_TYPE_MONO8);
    } else {
        channel = 3;
        CameraSetIspOutFormat(hCamera, CAMERA_MEDIA_TYPE_BGR8);
    }
 
    isInitialized = true;
    return true;
}
 
void CameraWrapper::close() {
    if (isInitialized) {
        CameraUnInit(hCamera);
        isInitialized = false;
    }
 
    if (g_pRgbBuffer != nullptr) {
        free(g_pRgbBuffer);
        g_pRgbBuffer = nullptr;
    }
 
    hCamera = -1;
}
 
bool CameraWrapper::getFrame(cv::Mat& frame, int timeout) {
    if (!isInitialized) {
        printf("Camera not initialized!\n");
        return false;
    }
 
    tSdkFrameHead sFrameInfo;
    BYTE* pbyBuffer = nullptr;
 
    // 获取图像缓冲区
    int status = CameraGetImageBuffer(hCamera, &sFrameInfo, &pbyBuffer, timeout);
    if (status != CAMERA_STATUS_SUCCESS) {
        // 超时属于正常情况，不打印错误信息
        if (status != CAMERA_STATUS_TIME_OUT) {
            printf("CameraGetImageBuffer failed, status: %d\n", status);
        }
        return false;
    }
 
    // 处理图像
    CameraImageProcess(hCamera, pbyBuffer, g_pRgbBuffer, &sFrameInfo);
 
    // 转换为OpenCV格式
    int type = (sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8) ? CV_8UC1 : CV_8UC3;
    frame = cv::Mat(
        cv::Size(sFrameInfo.iWidth, sFrameInfo.iHeight),
        type,
        g_pRgbBuffer
    ).clone();  // 使用clone确保数据被复制，避免悬空指针问题
 
    // 释放缓冲区
    CameraReleaseImageBuffer(hCamera, pbyBuffer);
 
    return !frame.empty();
}

bool CameraWrapper::setExposure(float exposureMs) {
    if (!isInitialized) return false;

    CameraSetAeState(hCamera, false);

    double exposureUs = exposureMs * 1000.0;  // 毫秒转微秒
    int status = CameraSetExposureTime(hCamera, exposureUs);
    return (status == CAMERA_STATUS_SUCCESS);
}