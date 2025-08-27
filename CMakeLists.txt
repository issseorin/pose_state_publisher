cmake_minimum_required(VERSION 3.8)
project(pose_state_publisher)

# ROS 2 빌드 도구
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# 노드에 필요한 의존성
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)

# [중요] 메시지 인터페이스 패키지를 의존성으로 추가
find_package(pose_state_publisher_interfaces REQUIRED)

# 이전 메시지 빌드 관련 코드는 모두 삭제해야 합니다.
# rosidl_generate_interfaces(...)
# find_package(rosidl_default_generators ...)
# ament_export_dependencies(rosidl_default_runtime)

ament_package()