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

include_directories(
  include
  include/test_camera
  ${OpenCV_INCLUDE_DIRS}) 

set(MVSDK_INCLUDE_DIR /home/aa/rm_camera/sdk/MDWS/include)
set(MVSDK_LIB_DIR /home/aa/rm_camera/sdk/MDWS/lib/x64)
include_directories(${MVSDK_INCLUDE_DIR})
link_directories(${MVSDK_LIB_DIR})


add_executable(image_publisher src/image_publisher.cpp src/Camera.cpp)
add_executable(visualizer_node src/visualizer_node.cpp)
add_executable(armor_detector_node src/armor_detector_node.cpp src/Pnp.cpp src/Camera.cpp src/KalmanFilter.cpp)
add_executable(kalman_node src/kalman_node.cpp src/KalmanFilter.cpp)
add_executable(serial_node src/serial_node.cpp)

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
target_link_libraries(armor_detector_node Eigen3::Eigen)
target_link_libraries(kalman_node Eigen3::Eigen)

install(TARGETS image_publisher visualizer_node armor_detector_node kalman_node serial_node
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
