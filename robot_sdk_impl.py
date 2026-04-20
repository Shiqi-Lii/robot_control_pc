from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any

import numpy as np

from .interfaces import CameraInterface, RobotInterface


class Ros2ImageCamera(CameraInterface):
    """Subscribe to one ROS 2 image topic and return RGB uint8 frames."""

    def __init__(self, topic: str, timeout_s: float = 3.0):
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise RuntimeError(
                "ROS 2 image dependencies are not available. "
                "Please source the ROS 2 environment before using image topics."
            ) from exc

        self._topic = topic
        self._timeout_s = timeout_s
        self._rclpy = rclpy
        self._executor = None
        self._executor_thread: threading.Thread | None = None
        self._node = None
        self._latest_image = None

        if not rclpy.ok():
            rclpy.init()

        class _CameraNode(Node):
            pass

        self._node = _CameraNode(f"gr00t_camera_{topic.strip('/').replace('/', '_')}")
        self._node.create_subscription(Image, topic, self._on_image, 10)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._executor_thread = threading.Thread(target=self._spin, daemon=True)
        self._executor_thread.start()
        self._wait_for_first_image()

    def _spin(self) -> None:
        while self._rclpy.ok():
            self._executor.spin_once(timeout_sec=0.1)

    def _on_image(self, msg) -> None:
        self._latest_image = msg

    def _wait_for_first_image(self) -> None:
        deadline = time.time() + self._timeout_s
        while self._latest_image is None and time.time() < deadline:
            time.sleep(0.05)
        if self._latest_image is None:
            raise TimeoutError(f"Timed out waiting for image topic {self._topic}")

    @staticmethod
    def _yuyv_to_rgb(data: np.ndarray, height: int, width: int) -> np.ndarray:
        yuyv = data.reshape(height, width // 2, 4).astype(np.float32)

        y0 = yuyv[:, :, 0]
        u = yuyv[:, :, 1]
        y1 = yuyv[:, :, 2]
        v = yuyv[:, :, 3]

        y = np.empty((height, width), dtype=np.float32)
        y[:, 0::2] = y0
        y[:, 1::2] = y1

        u_full = np.repeat(u[:, :, np.newaxis], 2, axis=2).reshape(height, width) - 128.0
        v_full = np.repeat(v[:, :, np.newaxis], 2, axis=2).reshape(height, width) - 128.0
        c = y - 16.0

        r = 1.164 * c + 1.596 * v_full
        g = 1.164 * c - 0.392 * u_full - 0.813 * v_full
        b = 1.164 * c + 2.017 * u_full

        rgb = np.stack([r, g, b], axis=-1)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def get_frame(self) -> np.ndarray:
        if self._latest_image is None:
            self._wait_for_first_image()

        msg = self._latest_image
        height = int(msg.height)
        width = int(msg.width)
        encoding = msg.encoding.lower()
        data = np.frombuffer(msg.data, dtype=np.uint8)

        if encoding == "rgb8":
            return data.reshape(height, width, 3).copy()
        if encoding == "bgr8":
            return data.reshape(height, width, 3)[:, :, ::-1].copy()
        if encoding == "mono8":
            mono = data.reshape(height, width)
            return np.repeat(mono[:, :, np.newaxis], 3, axis=2)
        if encoding in {"yuv422_yuy2", "yuyv", "yuy2"}:
            return self._yuyv_to_rgb(data, height, width)

        raise ValueError(
            f"Unsupported ROS image encoding '{msg.encoding}' on topic {self._topic}. "
            "Currently supported: rgb8, bgr8, mono8, yuv422_yuy2/yuyv."
        )


@dataclass
class YsRos2Config:
    observation_keys: list[str]
    action_keys: list[str]
    left_joint_names: list[str]
    right_joint_names: list[str]
    left_trajectory_topic: str
    right_trajectory_topic: str
    joint_state_topic: str = "/joint_states"
    left_gripper_uint8_enabled: bool = False
    left_gripper_cmd_topic: str = "/gripper_left/cmd"
    left_gripper_state_enabled: bool = False
    left_gripper_state_topic: str = "/gripper_left/state"
    left_gripper_open_cmd_value: int = 0
    left_gripper_closed_cmd_value: int = 1
    gripper_default_value: float = 0.5
    left_ee_pose_enabled: bool = False
    left_ee_pose_topic: str = "/planner/left_tcp_pose"
    right_ee_pose_enabled: bool = False
    right_ee_pose_topic: str = "/planner/right_tcp_pose"
    point_time_from_start: float = 0.1
    observation_timeout_s: float = 3.0
    reset_duration_s: float = 5.0
    reset_left_joint_positions: list[float] | None = None
    reset_right_joint_positions: list[float] | None = None
    # Optional EE delta (Servo) control path.
    ee_delta_frame_id: str = "base_link"
    ee_delta_source_hz: float = 30.0
    ee_delta_gain: float = 1.0
    ee_delta_command_in_type: str = "unitless"  # unitless or speed_units
    ee_delta_unitless_linear_scale: float = 0.4
    ee_delta_unitless_rotational_scale: float = 1.0
    left_delta_twist_topic: str = "/left_servo_node/delta_twist_cmds"
    right_delta_twist_topic: str = "/right_servo_node/delta_twist_cmds"
    left_start_servo_service: str = "/left_servo_node/start_servo"
    right_start_servo_service: str = "/right_servo_node/start_servo"
    auto_start_servo_for_ee_delta: bool = True


class YsRos2RobotController(RobotInterface):
    """Minimal YS ROS 2 controller: subscribe joint/gripper/ee, publish trajectory."""

    def __init__(self, config: YsRos2Config):
        self.config = config
        self._rclpy = None
        self._executor = None
        self._executor_thread: threading.Thread | None = None
        self._node = None
        self._left_trajectory_pub = None
        self._right_trajectory_pub = None
        self._left_gripper_uint8_cmd_pub = None
        self._left_delta_twist_pub = None
        self._right_delta_twist_pub = None
        self._latest_joint_state = None
        self._latest_left_gripper_uint8_state = None
        self._latest_left_ee_pose = None
        self._latest_right_ee_pose = None
        self._wants_left_ee_delta = False
        self._wants_right_ee_delta = False

    def connect(self) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import PoseStamped, TwistStamped
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from sensor_msgs.msg import JointState
            from std_msgs.msg import UInt8
            from std_srvs.srv import Trigger
            from trajectory_msgs.msg import JointTrajectory
        except ImportError as exc:
            raise RuntimeError(
                "ROS 2 dependencies are not available. "
                "Please source your robot workspace before running."
            ) from exc

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()

        class _RobotNode(Node):
            pass

        self._node = _RobotNode("gr00t_robot_control_pc")
        self._left_trajectory_pub = self._node.create_publisher(
            JointTrajectory,
            self.config.left_trajectory_topic,
            10,
        )
        self._right_trajectory_pub = self._node.create_publisher(
            JointTrajectory,
            self.config.right_trajectory_topic,
            10,
        )
        if self.config.left_gripper_uint8_enabled:
            self._left_gripper_uint8_cmd_pub = self._node.create_publisher(
                UInt8,
                self.config.left_gripper_cmd_topic,
                10,
            )

        self._wants_left_ee_delta = "delta_left_ee_pose" in self.config.action_keys
        self._wants_right_ee_delta = "delta_right_ee_pose" in self.config.action_keys
        if self._wants_left_ee_delta:
            self._left_delta_twist_pub = self._node.create_publisher(
                TwistStamped,
                self.config.left_delta_twist_topic,
                20,
            )
        if self._wants_right_ee_delta:
            self._right_delta_twist_pub = self._node.create_publisher(
                TwistStamped,
                self.config.right_delta_twist_topic,
                20,
            )

        self._node.create_subscription(
            JointState,
            self.config.joint_state_topic,
            self._on_joint_state,
            100,
        )
        if self.config.left_gripper_state_enabled:
            self._node.create_subscription(
                UInt8,
                self.config.left_gripper_state_topic,
                self._on_left_gripper_uint8_state,
                100,
            )
        if self.config.left_ee_pose_enabled:
            self._node.create_subscription(
                PoseStamped,
                self.config.left_ee_pose_topic,
                self._on_left_ee_pose,
                100,
            )
        if self.config.right_ee_pose_enabled:
            self._node.create_subscription(
                PoseStamped,
                self.config.right_ee_pose_topic,
                self._on_right_ee_pose,
                100,
            )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._executor_thread = threading.Thread(target=self._spin, daemon=True)
        self._executor_thread.start()

        print(
            "YS ROS2 controller connected with "
            f"joint_state_topic={self.config.joint_state_topic}, "
            f"left_gripper_uint8_cmd_topic={self.config.left_gripper_cmd_topic if self.config.left_gripper_uint8_enabled else 'disabled'}, "
            f"left_gripper_uint8_state_topic={self.config.left_gripper_state_topic if self.config.left_gripper_state_enabled else 'disabled'}, "
            f"left_ee_pose_topic={self.config.left_ee_pose_topic if self.config.left_ee_pose_enabled else 'disabled'}, "
            f"right_ee_pose_topic={self.config.right_ee_pose_topic if self.config.right_ee_pose_enabled else 'disabled'}"
        )
        if self._wants_left_ee_delta or self._wants_right_ee_delta:
            print(
                "EE delta control enabled for keys: "
                f"left={self._wants_left_ee_delta}, right={self._wants_right_ee_delta}, "
                f"command_in_type={self.config.ee_delta_command_in_type}"
            )
            if self.config.auto_start_servo_for_ee_delta:
                if self._wants_left_ee_delta:
                    self._try_start_servo(self.config.left_start_servo_service)
                if self._wants_right_ee_delta:
                    self._try_start_servo(self.config.right_start_servo_service)
        self._wait_for_first_joint_state()

    def disconnect(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
        if self._node is not None:
            self._node.destroy_node()
        if self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()
        print("YS ROS2 controller disconnected")

    def reset(self) -> None:
        reset_sent = False

        if self.config.reset_left_joint_positions:
            print(
                "Sending left arm reset trajectory with positions="
                f"{self.config.reset_left_joint_positions}"
            )
            self._publish_joint_trajectory(
                publisher=self._left_trajectory_pub,
                joint_names=self.config.left_joint_names,
                positions=np.asarray(self.config.reset_left_joint_positions, dtype=np.float64),
                duration_s=self.config.reset_duration_s,
            )
            reset_sent = True

        if self.config.reset_right_joint_positions:
            print(
                "Sending right arm reset trajectory with positions="
                f"{self.config.reset_right_joint_positions}"
            )
            self._publish_joint_trajectory(
                publisher=self._right_trajectory_pub,
                joint_names=self.config.right_joint_names,
                positions=np.asarray(self.config.reset_right_joint_positions, dtype=np.float64),
                duration_s=self.config.reset_duration_s,
            )
            reset_sent = True

        if not reset_sent:
            print("No reset joint positions configured. Keep current arm pose.")
            return

        print(f"Waiting {self.config.reset_duration_s:.2f}s for reset trajectory to finish...")
        time.sleep(max(self.config.reset_duration_s, 0.0))

    def _spin(self) -> None:
        while self._rclpy.ok():
            self._executor.spin_once(timeout_sec=0.1)

    def _on_joint_state(self, msg) -> None:
        self._latest_joint_state = msg

    def _on_left_gripper_uint8_state(self, msg) -> None:
        self._latest_left_gripper_uint8_state = msg

    def _on_left_ee_pose(self, msg) -> None:
        self._latest_left_ee_pose = msg

    def _on_right_ee_pose(self, msg) -> None:
        self._latest_right_ee_pose = msg

    def _wait_for_first_joint_state(self) -> None:
        deadline = time.time() + self.config.observation_timeout_s
        while self._latest_joint_state is None and time.time() < deadline:
            time.sleep(0.05)
        if self._latest_joint_state is None:
            raise TimeoutError(
                f"Timed out waiting for joint state topic {self.config.joint_state_topic}"
            )

    def _extract_named_positions(self, joint_names: list[str]) -> np.ndarray:
        msg = self._latest_joint_state
        name_to_index = {name: idx for idx, name in enumerate(msg.name)}
        missing = [name for name in joint_names if name not in name_to_index]
        if missing:
            raise KeyError(
                f"Joint names {missing} not found in {self.config.joint_state_topic}. "
                f"Available names: {list(msg.name)}"
            )
        return np.asarray(
            [msg.position[name_to_index[name]] for name in joint_names],
            dtype=np.float32,
        )

    def _extract_named_velocities(self, joint_names: list[str]) -> np.ndarray:
        msg = self._latest_joint_state
        name_to_index = {name: idx for idx, name in enumerate(msg.name)}
        velocities = []
        for name in joint_names:
            idx = name_to_index[name]
            velocities.append(msg.velocity[idx] if idx < len(msg.velocity) else 0.0)
        return np.asarray(velocities, dtype=np.float32)

    def _extract_gripper_value(self, side: str) -> np.ndarray:
        if side != "left":
            raise KeyError("Only left_gripper is supported in UInt8 bridge mode.")

        if side == "left" and self.config.left_gripper_state_enabled:
            if self._latest_left_gripper_uint8_state is None:
                return np.asarray([self.config.gripper_default_value], dtype=np.float32)
            raw_value = int(self._latest_left_gripper_uint8_state.data)
            if raw_value == self.config.left_gripper_open_cmd_value:
                normalized = float(self.config.left_gripper_open_cmd_value)
            elif raw_value == self.config.left_gripper_closed_cmd_value:
                normalized = float(self.config.left_gripper_closed_cmd_value)
            else:
                normalized = self.config.gripper_default_value
            return np.asarray([normalized], dtype=np.float32)

        return np.asarray([self.config.gripper_default_value], dtype=np.float32)

    def _extract_ee_pose(self, side: str) -> np.ndarray:
        latest_pose = self._latest_left_ee_pose if side == "left" else self._latest_right_ee_pose
        enabled = self.config.left_ee_pose_enabled if side == "left" else self.config.right_ee_pose_enabled
        if not enabled or latest_pose is None:
            return np.zeros(7, dtype=np.float32)
        pose = latest_pose.pose
        return np.asarray(
            [
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=np.float32,
        )

    def _try_start_servo(self, service_name: str) -> None:
        from std_srvs.srv import Trigger

        client = self._node.create_client(Trigger, service_name)
        if not client.wait_for_service(timeout_sec=2.0):
            print(f"[WARN] Servo start service not available: {service_name}")
            return
        future = client.call_async(Trigger.Request())
        deadline = time.time() + 3.0
        while time.time() < deadline and not future.done():
            time.sleep(0.05)
        if not future.done() or future.result() is None:
            print(f"[WARN] Failed to call Servo start service: {service_name}")
            return
        if future.result().success:
            print(f"[INFO] Servo started via {service_name}")
        else:
            print(f"[WARN] Servo start rejected by {service_name}: {future.result().message}")

    def _normalize_delta_ee_pose(self, value: Any, key_name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size < 6:
            raise ValueError(
                f"{key_name} expects at least 6 values [dx,dy,dz,drx,dry,drz], got {arr.size}"
            )
        return arr[:6]

    def _convert_delta_to_twist_cmd(self, delta6: np.ndarray) -> np.ndarray:
        # delta6 is per-step delta [dx, dy, dz, drx, dry, drz].
        source_hz = max(float(self.config.ee_delta_source_hz), 1e-9)
        gain = float(self.config.ee_delta_gain)
        mode = str(self.config.ee_delta_command_in_type).strip()

        if mode == "speed_units":
            return gain * np.concatenate([delta6[:3] * source_hz, delta6[3:] * source_hz], axis=0)
        if mode == "unitless":
            lin_scale = max(float(self.config.ee_delta_unitless_linear_scale), 1e-9)
            rot_scale = max(float(self.config.ee_delta_unitless_rotational_scale), 1e-9)
            lin = (delta6[:3] * source_hz) / lin_scale
            rot = (delta6[3:] * source_hz) / rot_scale
            return np.clip(gain * np.concatenate([lin, rot], axis=0), -1.0, 1.0)
        raise ValueError(
            f"Unsupported ee_delta_command_in_type='{mode}', expected 'unitless' or 'speed_units'"
        )

    def _publish_delta_twist(self, publisher, delta6: np.ndarray) -> None:
        from geometry_msgs.msg import TwistStamped

        cmd = self._convert_delta_to_twist_cmd(delta6)
        msg = TwistStamped()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self.config.ee_delta_frame_id
        msg.twist.linear.x = float(cmd[0])
        msg.twist.linear.y = float(cmd[1])
        msg.twist.linear.z = float(cmd[2])
        msg.twist.angular.x = float(cmd[3])
        msg.twist.angular.y = float(cmd[4])
        msg.twist.angular.z = float(cmd[5])
        publisher.publish(msg)

    def get_observation(self) -> dict[str, Any]:
        if self._latest_joint_state is None:
            self._wait_for_first_joint_state()

        observation: dict[str, Any] = {}
        for key in self.config.observation_keys:
            if key == "left_arm":
                observation[key] = self._extract_named_positions(self.config.left_joint_names)
            elif key == "left_arm_velocity":
                observation[key] = self._extract_named_velocities(self.config.left_joint_names)
            elif key == "right_arm":
                observation[key] = self._extract_named_positions(self.config.right_joint_names)
            elif key == "right_arm_velocity":
                observation[key] = self._extract_named_velocities(self.config.right_joint_names)
            elif key == "left_gripper":
                observation[key] = self._extract_gripper_value("left")
            elif key == "right_gripper":
                raise KeyError("right_gripper is not supported in UInt8 bridge mode.")
            elif key == "left_ee_pose":
                observation[key] = self._extract_ee_pose("left")
            elif key == "right_ee_pose":
                observation[key] = self._extract_ee_pose("right")
            else:
                raise KeyError(f"Unsupported observation key '{key}'")
        return observation

    def send_action(self, action: dict[str, Any]) -> None:
        # Keep original joint-space control path unchanged when EE delta keys are not provided.
        left_ee_delta = action.get("delta_left_ee_pose", None)
        right_ee_delta = action.get("delta_right_ee_pose", None)

        if left_ee_delta is not None and self._left_delta_twist_pub is not None:
            self._publish_delta_twist(
                self._left_delta_twist_pub,
                self._normalize_delta_ee_pose(left_ee_delta, "delta_left_ee_pose"),
            )
        elif "left_arm" in action:
            self._publish_joint_trajectory(
                publisher=self._left_trajectory_pub,
                joint_names=self.config.left_joint_names,
                positions=np.asarray(action["left_arm"], dtype=np.float64).reshape(-1),
            )

        if right_ee_delta is not None and self._right_delta_twist_pub is not None:
            self._publish_delta_twist(
                self._right_delta_twist_pub,
                self._normalize_delta_ee_pose(right_ee_delta, "delta_right_ee_pose"),
            )
        elif "right_arm" in action:
            self._publish_joint_trajectory(
                publisher=self._right_trajectory_pub,
                joint_names=self.config.right_joint_names,
                positions=np.asarray(action["right_arm"], dtype=np.float64).reshape(-1),
            )
        if "left_gripper" in action:
            left_value = float(np.asarray(action["left_gripper"]).reshape(-1)[0])
            self._publish_left_gripper_uint8_command(left_value)
        if "right_gripper" in action:
            raise KeyError("right_gripper action is not supported in UInt8 bridge mode.")

    def _publish_joint_trajectory(
        self,
        publisher,
        joint_names: list[str],
        positions: np.ndarray,
        duration_s: float | None = None,
    ) -> None:
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        if len(positions) != len(joint_names):
            raise ValueError(
                "Action length does not match configured joint names: "
                f"{len(positions)} vs {len(joint_names)}"
            )

        msg = JointTrajectory()
        msg.joint_names = list(joint_names)
        point = JointTrajectoryPoint()
        point.positions = positions.tolist()
        trajectory_duration = (
            self.config.point_time_from_start if duration_s is None else float(duration_s)
        )
        point.time_from_start = Duration(
            sec=int(trajectory_duration),
            nanosec=int((trajectory_duration % 1.0) * 1e9),
        )
        msg.points = [point]
        publisher.publish(msg)

    def _publish_left_gripper_uint8_command(self, normalized_value: float) -> None:
        from std_msgs.msg import UInt8

        if self._left_gripper_uint8_cmd_pub is None:
            return

        # Policy-to-command rule for left gripper bridge:
        # value > 0.5 -> close(1), otherwise open(0)
        raw_value = (
            self.config.left_gripper_closed_cmd_value
            if normalized_value > 0.5
            else self.config.left_gripper_open_cmd_value
        )
        msg = UInt8()
        msg.data = int(raw_value)
        self._left_gripper_uint8_cmd_pub.publish(msg)


def build_cameras_from_config(config: dict[str, Any]) -> dict[str, CameraInterface]:
    camera_keys = config.get("camera_keys", [])
    topics = config.get("camera_topics", {})
    timeout_s = float(config.get("camera_timeout_s", 3.0))
    return {
        camera_key: Ros2ImageCamera(topic=str(topics[camera_key]), timeout_s=timeout_s)
        for camera_key in camera_keys
    }


def build_robot_from_config(config: dict[str, Any]) -> RobotInterface:
    robot_cfg = YsRos2Config(
        observation_keys=list(config.get("state_keys", [])),
        action_keys=list(config.get("action_keys", [])),
        left_joint_names=list(config.get("left_joint_names", [])),
        right_joint_names=list(config.get("right_joint_names", [])),
        left_trajectory_topic=config.get(
            "left_trajectory_topic", "/arm_left_controller/joint_trajectory"
        ),
        right_trajectory_topic=config.get(
            "right_trajectory_topic", "/arm_right_controller/joint_trajectory"
        ),
        joint_state_topic=config.get("joint_state_topic", "/joint_states"),
        left_gripper_uint8_enabled=bool(config.get("left_gripper_uint8_enabled", False)),
        left_gripper_cmd_topic=str(config.get("left_gripper_cmd_topic", "/gripper_left/cmd")),
        left_gripper_state_enabled=bool(config.get("left_gripper_state_enabled", False)),
        left_gripper_state_topic=str(config.get("left_gripper_state_topic", "/gripper_left/state")),
        left_gripper_open_cmd_value=int(config.get("left_gripper_open_cmd_value", 0)),
        left_gripper_closed_cmd_value=int(config.get("left_gripper_closed_cmd_value", 1)),
        gripper_default_value=float(config.get("gripper_default_value", 0.5)),
        left_ee_pose_enabled=config.get("left_ee_pose_enabled", False),
        left_ee_pose_topic=config.get("left_ee_pose_topic", "/planner/left_tcp_pose"),
        right_ee_pose_enabled=config.get("right_ee_pose_enabled", False),
        right_ee_pose_topic=config.get("right_ee_pose_topic", "/planner/right_tcp_pose"),
        point_time_from_start=float(config.get("point_time_from_start", 0.1)),
        observation_timeout_s=float(config.get("observation_timeout_s", 3.0)),
        reset_duration_s=float(config.get("reset_duration_s", 5.0)),
        reset_left_joint_positions=config.get("reset_left_joint_positions"),
        reset_right_joint_positions=config.get("reset_right_joint_positions"),
        ee_delta_frame_id=str(config.get("ee_delta_frame_id", "base_link")),
        ee_delta_source_hz=float(config.get("ee_delta_source_hz", 30.0)),
        ee_delta_gain=float(config.get("ee_delta_gain", 1.0)),
        ee_delta_command_in_type=str(config.get("ee_delta_command_in_type", "unitless")),
        ee_delta_unitless_linear_scale=float(config.get("ee_delta_unitless_linear_scale", 0.4)),
        ee_delta_unitless_rotational_scale=float(config.get("ee_delta_unitless_rotational_scale", 1.0)),
        left_delta_twist_topic=str(config.get("left_delta_twist_topic", "/left_servo_node/delta_twist_cmds")),
        right_delta_twist_topic=str(config.get("right_delta_twist_topic", "/right_servo_node/delta_twist_cmds")),
        left_start_servo_service=str(config.get("left_start_servo_service", "/left_servo_node/start_servo")),
        right_start_servo_service=str(config.get("right_start_servo_service", "/right_servo_node/start_servo")),
        auto_start_servo_for_ee_delta=bool(config.get("auto_start_servo_for_ee_delta", True)),
    )
    return YsRos2RobotController(robot_cfg)
