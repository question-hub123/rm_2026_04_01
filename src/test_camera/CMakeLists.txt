cmake_minimum_required(VERSION 3.8)
project(test_camera)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(std_msgs REQUIRED)
find_package(armor_interfaces REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(serial_driver REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui ml)

include_directories(/home/aa/onnxruntime-linux-x64-1.16.1/include)

include_directories(
  include
  include/test_camera
  ${OpenCV_INCLUDE_DIRS}) 

set(ONNXRUNTIME_LIB_DIR "/home/aa/onnxruntime-linux-x64-1.16.1/lib")

# 告诉 CMake 在链接时使用这个路径
link_directories(${ONNXRUNTIME_LIB_DIR})

# 关键：设置可执行文件的 RPATH，将自定义库路径嵌入进去
# 方式一：针对所有目标设置全局 RPATH
set(CMAKE_INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}")
set(CMAKE_BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

set(MVSDK_INCLUDE_DIR /home/aa/rm_camera/sdk/MDWS/include)
set(MVSDK_LIB_DIR /home/aa/rm_camera/sdk/MDWS/lib/x64)
include_directories(${MVSDK_INCLUDE_DIR})
link_directories(${MVSDK_LIB_DIR})


add_executable(image_publisher src/image_publisher.cpp src/Camera.cpp)
add_executable(visualizer_node src/visualizer_node.cpp)
add_executable(armor_detector_node src/armor_detector_node.cpp src/Pnp.cpp src/Camera.cpp)

add_executable(kalman_node src/kalman_node.cpp src/KalmanFilter.cpp src/AngleKalman.cpp)
add_executable(serial_node src/serial_node.cpp)
add_executable(yolo_armor_detect
  src/yolo_armor_detect.cpp
  src/YoloPose.cpp     # <--- 必须加上这个！
  src/Camera.cpp       # <--- 如果你没加，也要加上！
  src/Pnp.cpp          # <--- PnP解算文件也要加上！
)


target_include_directories(image_publisher PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)


target_compile_features(image_publisher PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  image_publisher
  "rclcpp"
  "OpenCV"
  "cv_bridge"
  "image_transport"
  "std_msgs"
)

ament_target_dependencies(
  visualizer_node
  "rclcpp"
  "OpenCV"
  "cv_bridge"
  "image_transport"
  "std_msgs"
  "armor_interfaces"
)

ament_target_dependencies(
  armor_detector_node
  "rclcpp"
  "OpenCV"
  "cv_bridge"
  "image_transport"
  "std_msgs"
  "armor_interfaces"
)

ament_target_dependencies(
  yolo_armor_detect
  "rclcpp"
  "OpenCV"
  "cv_bridge"
  "image_transport"
  "std_msgs"
  "armor_interfaces"
)

ament_target_dependencies(
  kalman_node
  "rclcpp"
  "OpenCV"
  "cv_bridge"
  "image_transport"
  "std_msgs"
  "armor_interfaces"
)

ament_target_dependencies(
  serial_node
  "rclcpp"
  "OpenCV"
  "cv_bridge"
  "image_transport"
  "std_msgs"
  "armor_interfaces"
  "serial_driver"
)

target_link_libraries(image_publisher libMVSDK.so)
target_link_libraries(armor_detector_node libMVSDK.so)
target_link_libraries(yolo_armor_detect libMVSDK.so)
target_link_libraries(armor_detector_node Eigen3::Eigen)
target_link_libraries(kalman_node Eigen3::Eigen)
target_link_libraries(armor_detector_node ${OpenCV_LIBRARIES})

# 原有 ONNX Runtime 链接 (通过 link_directories)
target_link_libraries(yolo_armor_detect
  ${OpenCV_LIBS}
  onnxruntime   # ★ 只写库名，CMake 会自动去 link_directories 里找
)

# ===== 新增：强制找到 onnxruntime 库并显式链接 =====
find_library(ONNXRUNTIME_LIB onnxruntime
             PATHS ${ONNXRUNTIME_LIB_DIR}
             NO_DEFAULT_PATH
             REQUIRED)
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}")
# 显式链接找到的库 (避免 link_directories 失效)
target_link_libraries(yolo_armor_detect ${ONNXRUNTIME_LIB})
# 可能需要的系统依赖 (与现有链接不冲突)
target_link_libraries(yolo_armor_detect pthread dl)
# ===== 新增结束 =====

install(TARGETS image_publisher visualizer_node armor_detector_node kalman_node serial_node yolo_armor_detect
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()