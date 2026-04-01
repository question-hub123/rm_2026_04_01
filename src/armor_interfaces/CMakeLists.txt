cmake_minimum_required(VERSION 3.8)
project(armor_interfaces)

# 默认编译选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找必要的包
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)   # 关键：消息生成器
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# 生成接口（根据你的实际消息文件名调整）
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Armor.msg"
  "msg/ArmorArray.msg"
  DEPENDENCIES std_msgs geometry_msgs   # 如果消息中使用了 geometry_msgs 类型
)

# 测试（可保留）
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()