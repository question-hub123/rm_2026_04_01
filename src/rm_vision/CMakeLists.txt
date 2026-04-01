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

set(DAHENG_INCLUDE_DIR /home/aa/rm_camera/sdk/Galaxy_camera/inc)
set(DAHENG_LIB_DIR /home/aa/rm_camera/sdk/Galaxy_camera/lib/x86_64)
include_directories(${DAHENG_INCLUDE_DIR})
link_directories(${DAHENG_LIB_DIR})

add_executable(armor_detector_node src/armor_detector_node.cpp src/Camera.cpp src/Pnp.cpp)
add_executable(kalman_node src/kalman_node.cpp src/kalmanfilter.cpp)
add_executable(serial_node src/serial_node.cpp )

target_include_directories(armor_detector_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(armor_detector_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
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

target_link_libraries(armor_detector_node
  libgxiapi.so          # 注意是小写！
  liblog4cplus_gx.so    # 这是日志库，也加上
  # 如果找到了图像处理库，也加在这里，比如 libdximageproc.so
)

install(TARGETS armor_detector_node kalman_node serial_node
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

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
