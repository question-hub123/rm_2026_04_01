cmake_minimum_required(VERSION 3.8)
project(rm_vision)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(armor_interfaces REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(serial_driver REQUIRED)

include_directories(
  include
  include/rm_vision
  ${OpenCV_INCLUDE_DIRS}
)

# 【关键】设置大恒SDK路径
set(DAHENG_SDK_PATH "/home/aa/rm_camera/sdk/Galaxy_camera")
set(DAHENG_INCLUDE_DIR "${DAHENG_SDK_PATH}/inc")
set(DAHENG_LIB_DIR "${DAHENG_SDK_PATH}/lib/x86_64")

include_directories(${DAHENG_INCLUDE_DIR})
include_directories(
  ${DAHENG_INCLUDE_DIR}
  "/home/aa/rm_camera/sdk/Galaxy_camera/sample/GxSingleCamColor"
)

# 【关键】直接定义库文件的绝对路径
set(DAHENG_LIBS
  "${DAHENG_LIB_DIR}/libgxiapi.so"
  "${DAHENG_LIB_DIR}/liblog4cplus_gx.so"
)

add_executable(armor_detector_node src/armor_detector_node.cpp src/Camera.cpp src/Pnp.cpp)
add_executable(kalman_node src/kalman_node.cpp src/kalmanfilter.cpp)
add_executable(serial_node src/serial_node.cpp )

target_include_directories(armor_detector_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(armor_detector_node PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(
  armor_detector_node
  "rclcpp"
  "OpenCV"
  "armor_interfaces"
)

ament_target_dependencies(
  kalman_node
  "rclcpp"
  "OpenCV"
  "armor_interfaces"
)

ament_target_dependencies(
  serial_node
  "rclcpp"
  "OpenCV"
  "armor_interfaces"
  "serial_driver"
)

# 【关键】直接链接绝对路径的库文件
target_link_libraries(armor_detector_node
  ${DAHENG_LIBS}
  pthread
  ${OpenCV_LIBS}
)

install(TARGETS armor_detector_node kalman_node serial_node
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
