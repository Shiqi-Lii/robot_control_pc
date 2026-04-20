from __future__ import annotations

import argparse
from pathlib import Path

from .core import Gr00tRobotControlClient, RuntimeConfig
from .robot_sdk_impl import build_cameras_from_config, build_robot_from_config


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read the robot control config. "
            "Install it with: python -m pip install pyyaml or uv pip install pyyaml"
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at the top level")
    return data


def build_client_from_config(config: dict) -> Gr00tRobotControlClient:
    runtime = RuntimeConfig(
        policy_host=config["policy_host"],
        policy_port=config["policy_port"],
        timeout_ms=config.get("timeout_ms", 15000),
        control_fps=config.get("control_fps", 30.0),
        execution_mode=config.get("execution_mode", "async_queue"),
        open_loop_horizon=config.get("open_loop_horizon", 8),
        action_refill_threshold=config.get("action_refill_threshold", 2),
        rtc_overlap=config.get("rtc_overlap", 8),
        rtc_frozen=config.get("rtc_frozen", 4),
        max_steps=config.get("max_steps", 0),
        language_instruction=config.get("language_instruction", "pick up the object"),
        reset_on_start=config.get("reset_on_start", False),
    )

    robot = build_robot_from_config(config)
    cameras = build_cameras_from_config(config)

    return Gr00tRobotControlClient(
        runtime=runtime,
        robot=robot,
        cameras=cameras,
        camera_keys=config["camera_keys"],
        state_keys=config["state_keys"],
        action_keys=config.get("action_keys"),
        language_key=config.get("language_key", "annotation.human.task_description"),
        api_token=config.get("api_token"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robot-side control client for a remote GR00T policy server."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("robot_control_pc/config/example_config.yaml"),
        help="Path to the robot control config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading robot control config from: {args.config}")
    config = _load_yaml(args.config)
    print("Building robot-side GR00T client...")
    client = build_client_from_config(config)
    print("Starting robot-side GR00T client...")
    client.run()


if __name__ == "__main__":
    main()
