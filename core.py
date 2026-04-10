from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Any

import numpy as np

from gr00t.data.types import ModalityConfig
from gr00t.policy.server_client import PolicyClient

from .interfaces import CameraInterface, RobotInterface


def _infer_history_len(config: ModalityConfig | None) -> int:
    if config is None:
        return 1
    return max(len(config.delta_indices), 1)


def _format_value(value: Any) -> str:
    array = np.asarray(value)
    return np.array2string(array, precision=4, suppress_small=True, threshold=32)


def _format_mapping(mapping: dict[str, Any]) -> str:
    parts = [f"{key}={_format_value(value)}" for key, value in mapping.items()]
    return ", ".join(parts)


class TemporalBuffer:
    """Maintains fixed-length history for cameras and states."""

    def __init__(self, camera_history: int, state_history: int):
        self.camera_history = max(camera_history, 1)
        self.state_history = max(state_history, 1)
        self.camera_buffers: dict[str, deque[np.ndarray]] = {}
        self.state_buffers: dict[str, deque[np.ndarray]] = {}

    def push_camera(self, key: str, frame: np.ndarray) -> None:
        if key not in self.camera_buffers:
            self.camera_buffers[key] = deque(maxlen=self.camera_history)
        self.camera_buffers[key].append(frame)

    def push_state(self, key: str, value: np.ndarray) -> None:
        if key not in self.state_buffers:
            self.state_buffers[key] = deque(maxlen=self.state_history)
        self.state_buffers[key].append(value.astype(np.float32, copy=False))

    def get_camera_stack(self, key: str) -> np.ndarray:
        frames = list(self.camera_buffers[key])
        if not frames:
            raise ValueError(f"No camera frames buffered for key '{key}'")
        while len(frames) < self.camera_history:
            frames.insert(0, frames[0])
        return np.stack(frames[-self.camera_history :], axis=0)

    def get_state_stack(self, key: str) -> np.ndarray:
        values = list(self.state_buffers[key])
        if not values:
            raise ValueError(f"No state values buffered for key '{key}'")
        while len(values) < self.state_history:
            values.insert(0, values[0])
        return np.stack(values[-self.state_history :], axis=0).astype(np.float32, copy=False)


@dataclass
class RuntimeConfig:
    policy_host: str = "127.0.0.1"
    policy_port: int = 5555
    timeout_ms: int = 15000
    control_fps: float = 30.0
    open_loop_horizon: int = 8
    action_refill_threshold: int = 2  # Refill chunk when queued actions <= this value
    max_steps: int = 0
    language_instruction: str = "pick up the object"
    reset_on_start: bool = False


class Gr00tRobotControlClient:
    """
    Generic robot-side GR00T client.

    The robot SDK integration is provided through:
    1. RobotInterface.get_observation()
    2. a camera mapping
    3. user-supplied key configuration
    """

    def __init__(
        self,
        *,
        runtime: RuntimeConfig,
        robot: RobotInterface,
        cameras: dict[str, CameraInterface],
        camera_keys: list[str],
        state_keys: list[str],
        action_keys: list[str] | None = None,
        language_key: str = "annotation.human.task_description",
        api_token: str | None = None,
    ):
        self.runtime = runtime
        self.robot = robot
        self.cameras = cameras
        self.camera_keys = camera_keys
        self.state_keys = state_keys
        self.action_keys = action_keys
        self.language_key = language_key
        self._api_token = api_token
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_actions: list[dict[str, Any]] | None = None
        self._prefetch_error: Exception | None = None
        print(
            "Initializing PolicyClient "
            f"for tcp://{runtime.policy_host}:{runtime.policy_port} "
            f"(timeout={runtime.timeout_ms}ms)"
        )
        self.policy = PolicyClient(
            host=runtime.policy_host,
            port=runtime.policy_port,
            timeout_ms=runtime.timeout_ms,
            api_token=api_token,
            strict=False,
        )

        print("Requesting modality config from GR00T server...")
        modality_config = self.policy.get_modality_config()
        print("Received modality config from GR00T server.")
        video_cfg = modality_config.get("video")
        state_cfg = modality_config.get("state")
        action_cfg = modality_config.get("action")

        self.buffer = TemporalBuffer(
            camera_history=_infer_history_len(video_cfg),
            state_history=_infer_history_len(state_cfg),
        )
        self.server_action_keys = action_cfg.modality_keys if action_cfg else []

        if self.action_keys is None:
            self.action_keys = list(self.server_action_keys)

        print(
            "Configured robot client with "
            f"camera_keys={self.camera_keys}, "
            f"state_keys={self.state_keys}, "
            f"action_keys={self.action_keys}"
        )

    def _make_policy_client(self) -> PolicyClient:
        return PolicyClient(
            host=self.runtime.policy_host,
            port=self.runtime.policy_port,
            timeout_ms=self.runtime.timeout_ms,
            api_token=self._api_token,
            strict=False,
        )

    def _read_current_observation(self) -> dict[str, Any]:
        robot_obs = dict(self.robot.get_observation())
        for camera_key in self.camera_keys:
            if camera_key not in self.cameras:
                raise KeyError(f"Missing camera implementation for '{camera_key}'")
            frame = self.cameras[camera_key].get_frame()
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            robot_obs[camera_key] = frame
        return robot_obs

    def _update_buffers(self, observation: dict[str, Any]) -> None:
        for camera_key in self.camera_keys:
            self.buffer.push_camera(camera_key, observation[camera_key])
        for state_key in self.state_keys:
            raw_value = observation[state_key]
            array_value = np.asarray(raw_value, dtype=np.float32)
            if array_value.ndim == 0:
                array_value = array_value.reshape(1)
            self.buffer.push_state(state_key, array_value)

    def build_policy_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        self._update_buffers(observation)

        policy_observation: dict[str, Any] = {
            "video": {},
            "state": {},
            "language": {self.language_key: [[self.runtime.language_instruction]]},
        }

        for camera_key in self.camera_keys:
            policy_observation["video"][camera_key] = self.buffer.get_camera_stack(camera_key)[
                np.newaxis, ...
            ]
        for state_key in self.state_keys:
            policy_observation["state"][state_key] = self.buffer.get_state_stack(state_key)[
                np.newaxis, ...
            ]
        return policy_observation

    def decode_action_chunk(self, action_chunk: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        if not self.action_keys:
            raise ValueError("No action keys configured for robot command decoding")

        horizon = min(
            action_chunk[action_key].shape[1]
            for action_key in self.action_keys
            if action_key in action_chunk
        )
        decoded_actions: list[dict[str, Any]] = []
        for t in range(horizon):
            decoded: dict[str, Any] = {}
            for action_key in self.action_keys:
                if action_key not in action_chunk:
                    raise KeyError(f"Server did not return action key '{action_key}'")
                value = np.asarray(action_chunk[action_key][0, t], dtype=np.float32).reshape(-1)
                decoded[action_key] = float(value[0]) if value.size == 1 else value.copy()
            decoded_actions.append(decoded)
        return decoded_actions

    def _fetch_action_chunk(
        self,
        policy_observation: dict[str, Any],
        *,
        policy: PolicyClient | None = None,
        log_prefix: str = "",
    ) -> list[dict[str, Any]]:
        print(
            f"{log_prefix}Sending state observation to GR00T server: "
            f"{_format_mapping(policy_observation['state'])}"
        )
        chunk_start = time.time()
        request_policy = self.policy if policy is None else policy
        action_chunk, _ = request_policy.get_action(policy_observation)
        actions = self.decode_action_chunk(action_chunk)
        chunk_latency_ms = (time.time() - chunk_start) * 1000.0
        print(
            f"{log_prefix}Decoded action chunk from GR00T server, "
            f"action length: {len(actions)}, latency={chunk_latency_ms:.1f}ms"
        )
        return actions

    @staticmethod
    def _action_candidates_for_state(state_key: str) -> list[str]:
        aliases = {
            "left_arm": ["left_arm", "arm_left"],
            "arm_left": ["arm_left", "left_arm"],
            "right_arm": ["right_arm", "arm_right"],
            "arm_right": ["arm_right", "right_arm"],
            "left_gripper": ["left_gripper"],
            "right_gripper": ["right_gripper"],
        }
        return aliases.get(state_key, [state_key])

    def _project_future_state_observation(
        self,
        policy_observation: dict[str, Any],
        queued_actions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not queued_actions:
            return policy_observation

        projected_state = {
            key: np.array(value, copy=True) for key, value in policy_observation["state"].items()
        }
        future_action = queued_actions[-1]

        for state_key, state_value in projected_state.items():
            if state_value.ndim < 3:
                continue
            for action_key in self._action_candidates_for_state(state_key):
                if action_key not in future_action:
                    continue
                action_value = np.asarray(future_action[action_key], dtype=np.float32).reshape(-1)
                if action_value.size != state_value.shape[-1]:
                    continue
                state_value[0, -1, :] = action_value
                break

        return {
            "video": policy_observation["video"],
            "state": projected_state,
            "language": policy_observation["language"],
        }

    def _start_async_prefetch(
        self,
        policy_observation: dict[str, Any],
        queued_actions: list[dict[str, Any]],
    ) -> None:
        with self._prefetch_lock:
            if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
                return
            self._prefetch_actions = None
            self._prefetch_error = None

        projected_observation = self._project_future_state_observation(
            policy_observation,
            queued_actions,
        )

        def _worker() -> None:
            worker_policy = self._make_policy_client()
            try:
                actions = self._fetch_action_chunk(
                    projected_observation,
                    policy=worker_policy,
                    log_prefix="[async] ",
                )
                with self._prefetch_lock:
                    self._prefetch_actions = actions
            except Exception as exc:
                with self._prefetch_lock:
                    self._prefetch_error = exc

        thread = threading.Thread(target=_worker, daemon=True)
        with self._prefetch_lock:
            self._prefetch_thread = thread
        thread.start()

    def _collect_async_prefetch(self) -> list[dict[str, Any]] | None:
        with self._prefetch_lock:
            thread = self._prefetch_thread
        if thread is None or thread.is_alive():
            return None

        thread.join()
        with self._prefetch_lock:
            self._prefetch_thread = None
            actions = self._prefetch_actions
            error = self._prefetch_error
            self._prefetch_actions = None
            self._prefetch_error = None

        if error is not None:
            raise RuntimeError("Asynchronous action prefetch failed") from error
        return actions

    def run(self) -> None:
        print("Connecting to robot controller...")
        self.robot.connect()
        if self.runtime.reset_on_start:
            print("Resetting robot controller...")
            self.robot.reset()
        else:
            print("Skipping robot reset on start.")

        print(
            "Pinging GR00T server at "
            f"tcp://{self.runtime.policy_host}:{self.runtime.policy_port}..."
        )
        if not self.policy.ping():
            raise RuntimeError(
                f"Cannot reach GR00T server at {self.runtime.policy_host}:{self.runtime.policy_port}"
            )
        print("GR00T server is reachable. Entering control loop.")

        step = 0
        target_period = 1.0 / max(self.runtime.control_fps, 1e-6)
        action_queue: deque[dict[str, Any]] = deque()
        last_action: dict[str, Any] | None = None
        refill_threshold = max(
            0,
            min(self.runtime.action_refill_threshold, max(self.runtime.open_loop_horizon - 1, 0)),
        )

        try:
            # Prime buffers and first chunk
            first_obs = self._read_current_observation()
            first_policy_observation = self.build_policy_observation(first_obs)
            first_actions = self._fetch_action_chunk(first_policy_observation)
            for action in first_actions[: self.runtime.open_loop_horizon]:
                action_queue.append(action)
            if not action_queue:
                raise RuntimeError("Policy returned empty action chunk.")

            # Start prefetching the next chunk immediately, using queued actions to
            # approximate the future state at which the next chunk will start.
            seed_obs = self._read_current_observation()
            seed_policy_observation = self.build_policy_observation(seed_obs)
            self._start_async_prefetch(seed_policy_observation, list(action_queue))

            while self.runtime.max_steps <= 0 or step < self.runtime.max_steps:
                tic = time.time()

                prefetched_actions = self._collect_async_prefetch()
                if prefetched_actions:
                    for action in prefetched_actions[: self.runtime.open_loop_horizon]:
                        action_queue.append(action)

                if len(action_queue) <= refill_threshold:
                    with self._prefetch_lock:
                        inflight = self._prefetch_thread is not None and self._prefetch_thread.is_alive()
                    if not inflight:
                        obs = self._read_current_observation()
                        policy_observation = self.build_policy_observation(obs)
                        self._start_async_prefetch(policy_observation, list(action_queue))

                if action_queue:
                    action = action_queue.popleft()
                    last_action = action
                    print(f"Executing action[{step}]: {_format_mapping(action)}")
                    self.robot.send_action(action)
                    step += 1
                elif last_action is not None:
                    # Graceful fallback while waiting for the next async chunk.
                    print(f"Executing hold action[{step}]: {_format_mapping(last_action)}")
                    self.robot.send_action(last_action)
                    step += 1
                else:
                    raise RuntimeError("No action available to execute.")

                if 0 < self.runtime.max_steps <= step:
                    break

                elapsed = time.time() - tic
                if elapsed < target_period:
                    time.sleep(target_period - elapsed)
        finally:
            self.robot.disconnect()
