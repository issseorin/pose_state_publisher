## 주요기능
- MediaPipe Pose(multi) 기반 **손 든 사람** 탐지
- 가장 가까운 (화면점유율최대) 손 든 사람 **선택 및 추적**
- SEARCHING -> HANDUP_DETECT -> APPROACHING -> ARRIVE 상태 발행
- 바운딩박스, 스켈레톤, 중심점, 상태 라벨 시각화

### 빌드
```
cd ~/ros2_ws
colcon build --packages-select pose_state_publisher
source install/setup.bash
```
### 실행
```
# 카메라가 연결된 상태에서
ros2 run pose_state_publisher pose_state_publish
```
### 토픽 
```
ros2 topic echo /pose_state   # 상태 코드(Int32)
ros2 topic echo /x_offset     # x 픽셀 편차(Float32)
```
