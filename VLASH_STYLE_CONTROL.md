# VLASH-Style Control Loop

本文档说明 [core.py](/home/f/Isaac-GR00T/robot_control_pc/core.py) 当前采用的控制思路。它不是对 `mit-han-lab/vlash` 的完整复现，而是一个面向现有 GR00T server-client 架构的简化实现，核心借鉴了两点：

- 异步预取下一段 action chunk
- 用待执行动作对下一次推理输入做 future-state-aware 近似前滚

## 目标

相较于传统同步控制：

1. 读当前观测
2. 阻塞等待 `get_action()`
3. 执行动作
4. 再读观测

当前实现希望达到：

- 执行动作时，后台同时向 server 预取下一段动作
- 减少因为推理阻塞造成的动作停顿
- 在发起下一次推理时，尽量考虑“机器人在推理完成前还会继续执行旧动作”这一事实

## 整体结构

主流程位于 [Gr00tRobotControlClient.run()](/home/f/Isaac-GR00T/robot_control_pc/core.py#L339)。

控制循环涉及 5 个核心模块：

- [TemporalBuffer](/home/f/Isaac-GR00T/robot_control_pc/core.py#L33)
- [build_policy_observation()](/home/f/Isaac-GR00T/robot_control_pc/core.py#L182)
- [decode_action_chunk()](/home/f/Isaac-GR00T/robot_control_pc/core.py#L201)
- [_start_async_prefetch()](/home/f/Isaac-GR00T/robot_control_pc/core.py#L286)
- [_project_future_state_observation()](/home/f/Isaac-GR00T/robot_control_pc/core.py#L255)

## 执行流程

### 1. 启动阶段

启动后先完成：

- 连接机器人接口
- 根据配置决定是否 reset
- `ping` GR00T server
- 查询 `get_modality_config()`
- 根据 `delta_indices` 推断 observation 的时序长度

其中时序长度自动来自：

- `video.delta_indices`
- `state.delta_indices`

## 2. 首次同步拉取 action chunk

进入主循环前，会先做一次同步推理：

1. 读取当前 observation
2. 用 [build_policy_observation()](/home/f/Isaac-GR00T/robot_control_pc/core.py#L182) 打包成 `(B, T, ...)`
3. 调 [_fetch_action_chunk()](/home/f/Isaac-GR00T/robot_control_pc/core.py#L221)
4. 用 [decode_action_chunk()](/home/f/Isaac-GR00T/robot_control_pc/core.py#L201) 把 `(B, T, D)` 形式的 action 拆成逐步动作列表
5. 把前 `open_loop_horizon` 步放入本地 `action_queue`

这样做的目的是先让本地队列里有一段可执行动作，再进入实时循环。

## 3. 动作队列

`action_queue` 是当前控制循环的核心状态。

它保存“已经从 server 拿到，但还没真正发给机器人”的动作序列。主循环中每个控制周期只做一件事：

- 从队列头部取一条动作
- 调 `self.robot.send_action(action)` 下发到机器人

好处是：

- 动作执行节拍可以和推理解耦
- server 一次返回多步动作时，本地可以连续执行
- 即便下一段 chunk 还没回来，也不必马上停下

## 4. 异步预取下一段动作

当前实现通过 [\_start_async_prefetch()](/home/f/Isaac-GR00T/robot_control_pc/core.py#L286) 在后台线程中创建新的 `PolicyClient`，并调用 `get_action()`。

它的行为是：

1. 主线程继续按 `control_fps` 执行动作队列
2. 后台线程独立向 server 请求下一段 action chunk
3. 后台完成后，把结果放到：
   - `self._prefetch_actions`
   - 或 `self._prefetch_error`
4. 主线程在下一轮循环里用 [\_collect_async_prefetch()](/home/f/Isaac-GR00T/robot_control_pc/core.py#L321) 收取结果

这就是当前实现里最接近 VLASH “async inference” 的部分。

## 5. Future-State-Aware 近似前滚

后台预取下一段动作时，直接用“当前时刻 observation”去推理会有时间错位问题：

- 推理需要时间
- 推理期间机器人仍在继续执行旧动作
- 所以下一段动作真正开始执行时，机器人状态已经不是发起推理时的状态

当前实现用 [\_project_future_state_observation()](/home/f/Isaac-GR00T/robot_control_pc/core.py#L255) 做了一个简化近似：

1. 复制当前 `policy_observation["state"]`
2. 取 `queued_actions[-1]`，即当前队列中最后一条待执行动作
3. 用这条动作去覆盖每个 state stream 的最后一个时间步

当前只对“动作维度与状态维度一致”的键生效，例如：

- `left_arm` 或 `arm_left`
- `right_arm` 或 `arm_right`
- `left_gripper`
- `right_gripper`

这是一个非常轻量的 future-state-aware 近似：

- 它不做真实动力学预测
- 不根据推理延迟精确外推多步
- 只是把下一次推理输入向“将来会执行到的状态”推近一点

## 6. 何时触发下一次预取

控制循环中有一个阈值：

- `action_refill_threshold`

当 `len(action_queue) <= action_refill_threshold` 时，如果后台当前没有正在进行的预取，就会：

1. 再读取一次最新 observation
2. 更新 buffer
3. 启动一次新的异步预取

这样可以避免：

- 动作队列完全空掉后才开始请求
- 因等待 server 导致机器人停顿

## 7. 控制节拍

主循环执行频率由：

- `control_fps`

控制。实现方式是：

1. 记录本步开始时间
2. 执行动作发送
3. 计算本步已经消耗的时间
4. 若未达到目标周期，则 `sleep` 补足

因此当前实现属于：

- 软实时控制循环

而不是严格硬实时 RTC。

## 8. 关键配置参数

当前最重要的可调参数在 [example_config.yaml](/home/f/Isaac-GR00T/robot_control_pc/config/example_config.yaml)：

- `control_fps`
  - 动作执行频率
- `open_loop_horizon`
  - 每次从 chunk 中实际放入本地队列的步数
- `action_refill_threshold`
  - 队列剩余多少步时开始预取下一段
- `point_time_from_start`
  - 每个 `JointTrajectoryPoint` 的目标到达时间
- `timeout_ms`
  - 与 server 通信的超时时间
- `reset_on_start`
  - 启动时是否先回初始位姿

### 调参建议

如果关注稳定性：

- 降低 `control_fps`
- 增大 `point_time_from_start`
- 减小 `open_loop_horizon`

如果关注流畅性和减少停顿：

- 适当增大 `open_loop_horizon`
- 适当增大 `action_refill_threshold`

如果希望反应更快：

- 降低 `open_loop_horizon`
- 结合更低的推理延迟

## 9. 与原始同步版本的区别

旧版同步思路大致是：

1. 读 observation
2. 同步 `get_action()`
3. 执行前几步动作
4. 再读 observation

当前版本的区别在于：

- 引入了 `action_queue`
- 引入了后台线程异步预取
- 不再每个 chunk 都阻塞主循环
- 在预取时对 state 做了 future-state-aware 的近似前滚

## 10. 当前实现的局限

当前版本仍然不是完整 VLASH，主要局限有：

- future-state projection 很粗糙
  - 只使用 `queued_actions[-1]`
  - 没有根据真实推理延迟外推多步
- 只对“状态维度与动作维度一致”的键自然生效
- 还没有 action quantization
- 还没有更细的时间对齐和融合逻辑
- 仍然依赖 Python 线程与 `time.sleep()`，属于软实时

## 11. 适用场景

当前实现更适合：

- GR00T server 在另一台机器上
- 单次推理存在明显延迟
- 机器人希望保持连续动作而不是“推理一下停一下”
- 关节位置型 action 为主的控制任务

## 12. 一句话总结

当前 `robot_control_pc` 的控制循环可以概括为：

- 用本地队列持续执行动作
- 用后台线程异步拉取下一段 action chunk
- 用待执行动作对下一次推理的 state 做轻量 future-state-aware 前滚

这就是它与 VLASH 最接近的部分。
