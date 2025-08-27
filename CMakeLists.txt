# [ADDED] 새로 생성
cmake_minimum_required(VERSION 3.8)
project(pose_state_publisher)

# 메시지 인터페이스 생성을 위해 필요
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# PoseInfo.msg 인터페이스 생성
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PoseInfo.msg"
)

# 런타임 의존성 내보내기
ament_export_dependencies(rosidl_default_runtime)

ament_package()
