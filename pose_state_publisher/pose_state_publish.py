import cv2
import time
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from std_msgs.msg import Int32
from std_msgs.msg import Float32 

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ----- 전역 설정 및 상수 정의 -----
CAM_INDEX = 2
MODEL_PATH = "/home/seorin/mediapipe_ws/pose_landmarker_full.task"
MAX_PEOPLE = 5
DRAW_SKELETON = True
XALIGN_TOL_RATIO = 0.1   # 화면 중앙 정렬 허용 오차 (가로폭 비율)

# ----- 임계값 설정 -----
THRESH_AREA_FAR = 0.10
THRESH_AREA_NEAR = 0.50

# ----- Mediapipe 포즈 인덱스 -----
IDX = {
    "NOSE": 0,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
}

# ----- 상태 정의 -----
STATE_SEARCHING = 0
STATE_HANDUP_DETECT = 1
STATE_APPROACHING = 2
STATE_ARRIVE = 3


# ===== 손 든 사람 판단 =====
def hand_up_flag(landmarks_norm):
    try:
        ls_y = landmarks_norm[IDX["LEFT_SHOULDER"]].y
        rs_y = landmarks_norm[IDX["RIGHT_SHOULDER"]].y
        lw_y = landmarks_norm[IDX["LEFT_WRIST"]].y
        rw_y = landmarks_norm[IDX["RIGHT_WRIST"]].y
    except Exception:
        return False, None, None, None

    shoulder_y = min(ls_y, rs_y)
    is_up = (lw_y is not None and lw_y < shoulder_y) or (rw_y is not None and rw_y < shoulder_y)
    return is_up, shoulder_y, lw_y, rw_y


# ===== 상체 바운딩박스 (픽셀 좌표) =====
def landmarks_bbox_px(landmarks_norm, w, h, padding=0.08):
    UPPER_IDX = [
        IDX["NOSE"],
        IDX["LEFT_SHOULDER"], IDX["RIGHT_SHOULDER"],
        IDX["LEFT_ELBOW"], IDX["RIGHT_ELBOW"],
        IDX["LEFT_WRIST"], IDX["RIGHT_WRIST"],
    ]
    xs = [landmarks_norm[i].x for i in UPPER_IDX if landmarks_norm[i].x is not None]
    ys = [landmarks_norm[i].y for i in UPPER_IDX if landmarks_norm[i].y is not None]
    if not xs or not ys:
        return None

    x_min = max(0.0, min(xs) - padding)
    y_min = max(0.0, min(ys) - padding)
    x_max = min(1.0, max(xs) + padding)
    y_max = min(1.0, max(ys) + padding)

    x1 = int(x_min * w); y1 = int(y_min * h)
    x2 = int(x_max * w); y2 = int(y_max * h)

    x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


# ===== bbox로부터 x_offset(px) 계산 =====
def x_offset(bbox, frame_width):
    x1, _, x2, _ = bbox
    cx = (x1 + x2) / 2.0
    return float(cx - (frame_width / 2.0)) 

# ===== bbox의 면적 비율 계산 =====
def bbox_area_ratio(bbox, frame_area):
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    return area / frame_area


class PoseStatePublisher(Node):
    def __init__(self):
        super().__init__("pose_state_publisher")
        self.publisher_ = self.create_publisher(Int32, "pose_state", 10)
        self.xoff_pub = self.create_publisher(Float32, "x_offset", 10) 

        # 상태머신 변수
        self.state = STATE_SEARCHING
        self.target_landmark_index = -1

        # MediaPipe 세팅
        BaseOptions = python.BaseOptions
        PoseLandmarker = vision.PoseLandmarker
        PoseLandmarkerOptions = vision.PoseLandmarkerOptions
        RunningMode = vision.RunningMode

        base_opts = BaseOptions(model_asset_path=MODEL_PATH)
        options = PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=RunningMode.VIDEO,
            num_poses=MAX_PEOPLE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = PoseLandmarker.create_from_options(options)

        # 비디오 캡쳐
        self.cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error("카메라를 열 수 없습니다.")
            raise RuntimeError("카메라를 열 수 없습니다.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.ts_ms = 0
        self.timer = self.create_timer(0.033, self.loop_once)
        self.publish_state(self.state) # 최초 상태 송신

    def publish_state(self, state_value: int):
        msg = Int32()
        msg.data = state_value
        self.publisher_.publish(msg)
        state_name = {
            STATE_SEARCHING: "SEARCHING",
            STATE_HANDUP_DETECT: "HANDUP_DETECT",
            STATE_APPROACHING: "APPROACHING",
            STATE_ARRIVE: "ARRIVE",
        }.get(state_value, f"UNKNOWN({state_value})")
        self.get_logger().info(f"pose_state = {state_value} ({state_name})")

    def change_state(self, new_state: int):
        if new_state != self.state:
            self.state = new_state
            self.publish_state(self.state)

    def loop_once(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn("프레임을 읽지 못했습니다.")
            return

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_area = w * h

        # MediaPipe 호출
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self.ts_ms += 33
        result = self.detector.detect_for_video(mp_image, self.ts_ms)

        # 현재 프레임에서 '손 든 사람' 수집
        handup_people = []
        if result.pose_landmarks:
            for i, lm in enumerate(result.pose_landmarks):
                if hand_up_flag(lm)[0]:
                    handup_people.append((i, lm))

        # ===== 상태머신 =====
        if result.pose_landmarks:
            # STATE 0: 손 든 사람을 찾는 중
            if self.state == STATE_SEARCHING:
                if len(handup_people) > 0:
                    self.change_state(STATE_HANDUP_DETECT)

            # STATE 1: 손 든 사람 중 가장 가까운 사람(화면 내 면적 최대)을 선택
            elif self.state == STATE_HANDUP_DETECT:
                best_idx = -1
                best_area_ratio = -1.0
                for i, lm in handup_people:
                    bbox = landmarks_bbox_px(lm, w, h, padding=0.08)
                    ratio = bbox_area_ratio(bbox, frame_area)
                    if ratio > best_area_ratio:
                        best_idx = i
                        best_area_ratio = ratio

                if best_idx >= 0:
                    self.target_landmark_index = best_idx
                    self.change_state(STATE_APPROACHING)

            # STATE 2: 선택된 사람을 추적하며 중앙정렬+근접 여부 판단
            elif self.state == STATE_APPROACHING:
                if 0 <= self.target_landmark_index < len(result.pose_landmarks):
                    lm = result.pose_landmarks[self.target_landmark_index]
                    bbox = landmarks_bbox_px(lm, w, h, padding=0.08)
                    ratio = bbox_area_ratio(bbox, frame_area)
                    if bbox:
                        x1, y1, x2, y2 = bbox

                        # 시각화 (추적 대상: 빨간 박스)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, f"tracking ratio: {ratio:.2f}", (x1, max(0, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        # x 편차 계산 및 발행
                        x_offset_px = x_offset((x1, y1, x2, y2), w)
                        self.xoff_pub.publish(Float32(data=x_offset_px))

                        if abs(x_offset_px) < (w * XALIGN_TOL_RATIO) and ratio >= THRESH_AREA_NEAR:
                             self.change_state(STATE_ARRIVE)

                        # ---- 중앙점 표시( bbox 표시 중에만 ) ----
                        cx_i = int((x1 + x2) / 2.0); cy_i = int((y1 + y2) / 2.0)
                        cv2.drawMarker(frame, (cx_i, cy_i), (0, 0, 255),markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
                        cv2.drawMarker(frame, (w // 2, h // 2), (150, 150, 150),markerType=cv2.MARKER_TILTED_CROSS, markerSize=16, thickness=1)

            # STATE 3: 도착 완료 (필요 시 후속 로직 추가)
            elif self.state == STATE_ARRIVE:
                pass

        # ===== 시각화: 손 든 사람들의 bbox, 스켈레톤, 상태라벨 =====
        if result.pose_landmarks:
            for idx, lm in enumerate(result.pose_landmarks):
                is_up, _, _, _ = hand_up_flag(lm)
                if is_up:
                    bbox = landmarks_bbox_px(lm, w, h, padding=0.08)
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2) # 손 든 사람: 녹색 박스
                        label = "HAND UP" # 라벨(바깥, 좌측 상단)
                        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

                if DRAW_SKELETON:
                    def to_px(x, y): return (int(x * w), int(y * h))
                    try:
                        p_ls = to_px(lm[IDX["LEFT_SHOULDER"]].x, lm[IDX["LEFT_SHOULDER"]].y)
                        p_rs = to_px(lm[IDX["RIGHT_SHOULDER"]].x, lm[IDX["RIGHT_SHOULDER"]].y)
                        p_lw = to_px(lm[IDX["LEFT_WRIST"]].x, lm[IDX["LEFT_WRIST"]].y)
                        p_rw = to_px(lm[IDX["RIGHT_WRIST"]].x, lm[IDX["RIGHT_WRIST"]].y)
                        cv2.line(frame, p_ls, p_rs, (255, 200, 0), 2)
                        cv2.line(frame, p_ls, p_lw, (100, 255, 255), 1)
                        cv2.line(frame, p_rs, p_rw, (100, 255, 255), 1)
                        cv2.circle(frame, p_lw, 4, (200, 255, 200), -1)
                        cv2.circle(frame, p_rw, 4, (200, 255, 200), -1)
                    except Exception:
                        pass

        # 화면 좌측 상단에 현재 상태 표시
        state_text = {
            STATE_SEARCHING: "SEARCHING",
            STATE_HANDUP_DETECT: "HANDUP_DETECT",
            STATE_APPROACHING: "APPROACHING",
            STATE_ARRIVE: "ARRIVE",
        }.get(self.state, "UNKNOWN")
        cv2.putText(frame, f"STATE: {state_text}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 200, 255), 2)

        cv2.imshow("Pose State (ROS2)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            rclpy.shutdown()

    def destroy_node(self):
        # 리소스 정리
        try:
            if hasattr(self, "cap") and self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()
    node = PoseStatePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # rclpy.shutdown() 는 loop_once 에서 호출될 수 있으므로 안전하게 호출
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()