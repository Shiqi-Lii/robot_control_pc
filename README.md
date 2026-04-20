# Robot Control PC

提供了一套部署在机器人电脑上的精简版 GR00T `client`，目标是：

1. 从本机相机和机器人控制接口采集观测。
2. 将观测打包发给远端 GR00T `server`
3. 调用远端 GR00T `server` 获取 action chunk。
4. 将 action chunk 转成机器人 SDK 可执行的控制指令。

## 目录结构

- `main.py`: 控制端入口。
- `core.py`: GR00T `PolicyClient`、时序缓存、观测打包、动作解码、主循环。
- `interfaces.py`: 机器人和相机的最小抽象接口。
- `robot_sdk_impl.py`: 机器人侧 ROS 2 订阅/发布实现。
- `config/example_config.json`: 配置示例。
- `VLASH_STYLE_CONTROL.md`: 当前异步控制循环与 VLASH 风格思路的技术说明。

## 已接入的本机控制接口

当前程序只保留这几类要用的输入，并且按双臂组织：

state 相关：
- 关节状态：`/joint_states`
- 左夹爪状态：`/gripper_left/state`（`std_msgs/msg/UInt8`）
- 末端位姿：`/planner/left_tcp_pose` 或 `/planner/right_tcp_pose`
- 图像：`/image_raw` ROS 2 类型 `sensor_msgs/msg/Image`

action 相关：
- 控制下发：`/arm_left_controller/joint_trajectory` 和 `/arm_right_controller/joint_trajectory`
- 末端增量控制：`/left_servo_node/delta_twist_cmds` 和 `/right_servo_node/delta_twist_cmds`（可选）
- 左夹爪控制：`/gripper_left/cmd`（UInt8 bridge）


## 使用方式

先在 GPU 服务器上启动 GR00T 推理服务：

```bash
python gr00t/eval/run_gr00t_server.py \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path /path/to/checkpoint \
  --host 0.0.0.0 \
  --port 5555
```

然后在机器人电脑侧：

```bash
source /home/f/lib/ros2Control_ws/install/setup.bash
source /home/f/lib/moveit2_ws/install/setup.bash
source /home/f/ysrobot_ws2/common/install/setup.bash
source /home/f/ysrobot_ws2/handware/install/setup.bash
source /home/f/ysrobot_ws2/robot/ys_dual_arm/install/setup.bash
python -m robot_control_pc.main --config robot_control_pc/config/example_config.yaml 
```

## 需要替换的部分

### 1. 机器人接口

`robot_sdk_impl.py` 里的 `YsRos2RobotController` 现在只做两件事：

- 订阅左右臂 joint / gripper / ee_pose
- 分别发布 `left_arm` / `right_arm` 到对应 `joint_trajectory`
- 把 `left_gripper` 发布到 `/gripper_left/cmd`（`UInt8`）

程序不会自动把所有状态都塞进 observation。
真正发给 GR00T server 的状态，由 [example_config.json] 里的 `state_keys` 决定。

### 2. 相机接口

当前相机实现只支持一种方式：

- 订阅 ROS 2 `sensor_msgs/msg/Image`

如果当前相机是通过 `v4l2_camera` 发布的，可以先这样启动：

```bash
sudo apt install v4l-utils
ros2 run v4l2_camera v4l2_camera_node --ros-args \
  -p video_device:=/dev/video6 \
  -p pixel_format:=YUYV \
  -p image_size:="[640,480]" \
  -p output_encoding:=yuv422_yuy2 \
  -r image_raw:=/top/image_raw
```

图像 topic 例如 `/image_raw`，在配置中这样填：

```json
"camera_topics": {
  "front": "/image_raw"
}
```

当前 `Ros2ImageCamera` 支持这些编码：

- `rgb8`
- `bgr8`
- `mono8`
- `yuv422_yuy2` / `yuyv`

### 3. 配置文件

修改 `config/example_config.json`：

- `policy_host` / `policy_port`: 远端 server 地址。
- `execution_mode`: `async_queue` 或 `rtc`
- `joint_state_topic`: 文档里明确可用的状态 topic 是 `/joint_states`
- `left_trajectory_topic` / `right_trajectory_topic`: 左右臂控制 topic
- `reset_on_start`: 启动时是否先执行一次回初始位姿
- `reset_duration_s`: 回初始位姿轨迹执行时长
- `reset_left_joint_positions` / `reset_right_joint_positions`: 回初始位姿的关节角
- `left_gripper_uint8_enabled`: 是否启用左夹爪 `UInt8` bridge 控制
- `left_gripper_cmd_topic`: 左夹爪 bridge 控制话题，默认 `/gripper_left/cmd`
- `left_gripper_state_enabled`: 是否订阅左夹爪 bridge 状态
- `left_gripper_state_topic`: 左夹爪 bridge 状态话题，默认 `/gripper_left/state`
- `left_ee_pose_topic` / `right_ee_pose_topic`: 左右 TCP 位姿 topic
- `left_joint_names` / `right_joint_names`: 要和 `joint_states` 里的关节名一致
- `camera_keys`: 必须与模型的 `video` modality key 对齐。
- `state_keys`: 必须与模型的 `state` modality key 对齐。
- `action_keys`: 必须与模型的 `action` modality key 对齐。
- `language_key`: 默认使用 `annotation.human.task_description`。

目前 `YsRos2RobotController.get_observation()` 可以提供这些状态键：

- `left_arm`
- `left_arm_velocity`
- `right_arm`
- `right_arm_velocity`
- `left_gripper`
- `left_ee_pose`
- `right_ee_pose`

`YsRos2RobotController.send_action()` 目前支持这些动作键：

- 关节控制（保持原逻辑）：
  - `left_arm`
  - `right_arm`
- 末端增量控制（新增，可选）：
  - `delta_left_ee_pose`：长度至少 6，格式 `[dx, dy, dz, drx, dry, drz]`
  - `delta_right_ee_pose`：长度至少 6，格式 `[dx, dy, dz, drx, dry, drz]`
- 夹爪控制：
  - `left_gripper`

控制切换规则（按每只手臂独立）：

- 若 action 中包含 `delta_left_ee_pose`，左臂优先走 Servo 增量控制；
- 否则若包含 `left_arm`，左臂走原有关节轨迹控制；
- 右臂同理（`delta_right_ee_pose` 优先于 `right_arm`）。

## 关于时序维度

`core.py` 在启动时会调用 `policy.get_modality_config()`，自动读取 server 侧模型需要的历史长度：

- `video.delta_indices`
- `state.delta_indices`

然后在机器人电脑本地维护时序缓存，自动把单帧观测组织成：

- 图像: `(B=1, T, H, W, C)`
- 状态: `(B=1, T, D)`
- 语言: `[[instruction]]`

所以只需要持续提供“当前时刻”的相机和机器人状态。

## Observation 如何自己决定

可以直接修改配置里的 `state_keys`，决定 observation 中包含哪些状态。

当前支持的状态键有：

- `left_arm`
- `left_arm_velocity`
- `right_arm`
- `right_arm_velocity`
- `left_gripper`
- `left_ee_pose`
- `right_ee_pose`

例如只传左臂关节和左夹爪：

```json
"state_keys": ["left_arm", "left_gripper"]
```

`YsRos2RobotController.get_observation()` 只会生成 `state_keys` 里列出来的键。

如果机器人控制接口需要：

- gripper 开关量二值化
- 动作平滑 / 限幅 / 安全检查

直接在 `core.py` 的 `decode_action_chunk()` 或 `robot_sdk_impl.py` 的 `send_action()` 中加入自己的逻辑。

## 与采集程序比对
对照 robdata_recorder_3.0` 的配置和代码，除了相机外，当前 `robot_control_pc` 采用的话题和字段获取方式与采集程序是对齐的：

- `joint_states`
  依据：
  [state_spec.yaml] [recorder_node.py]
  采集程序保存字段为 `name`, `q`, `dq`, `effort`。
  现在控制端也是按 `name -> position/velocity` 重排左右臂关节，这个是正确的。

- `ee_pose`
  依据：
  [state_spec.yaml]和 [recorder_node.py]
  采集程序分别记录：
  `/planner/left_tcp_pose`
  `/planner/right_tcp_pose`
  并提取：
  `pos=[x,y,z]`
  `quat=[x,y,z,w]`
  现在控制端输出 `left_ee_pose/right_ee_pose` 也是同样的 7 维结构。

## 夹爪控制
当前仅保留 `/home/f/ysrobot_ws2/run_gripper_global.sh` 对应的左夹爪 `UInt8` bridge：

- 下发命令：`/gripper_left/cmd`（`std_msgs/UInt8`）
- 读取状态：`/gripper_left/state`（`std_msgs/UInt8`）

动作映射规则：

- `left_gripper > 0.5` 时发送 `1`（close）
- `left_gripper <= 0.5` 时发送 `0`（open）
