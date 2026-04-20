# RTC-Style Control Loop

本文档保留的是一版 RTC / VLASH 风格控制思路说明，用来记录曾尝试的方向。

当前 [core.py](/home/f/Isaac-GR00T/robot_control_pc/core.py) 已经切回较稳定的“异步预取 + 动作队列补货”实现，不再使用本文件里描述的 `overlap / frozen` 切换逻辑。

如果后续需要再次切回 RTC 风格，可以把这份文档作为设计参考。

## 核心思路

当前实现把动作预测看成“重叠区补全”问题：

1. 当前 chunk 正在执行时，不等它跑完，就提前触发下一次推理
2. 新 chunk 与旧 chunk 在尾部保留一个重叠区
3. 在真正切换时：
   - `frozen` 这部分保持旧 chunk 的尾部不变
   - 剩余重叠区做软融合

这样可以减少 chunk 边界处的动作突变。

## 关键参数

参数来自 [example_config.yaml](/home/f/Isaac-GR00T/robot_control_pc/config/example_config.yaml)：

- `open_loop_horizon`
  - 当前每段 chunk 实际使用的长度
  - RTC 推荐至少 `32`
- `rtc_overlap`
  - 相邻 chunk 的重叠长度
- `rtc_frozen`
  - 切换时保留旧 chunk 尾部、不被新 chunk 覆盖的步数
- `control_fps`
  - 动作执行频率
- `point_time_from_start`
  - 单个 `JointTrajectoryPoint` 的目标到达时间

默认配置是：

- `open_loop_horizon: 32`
- `rtc_overlap: 8`
- `rtc_frozen: 4`

## 运行流程

主逻辑在 [run()]( /home/f/Isaac-GR00T/robot_control_pc/core.py#L399 )。

### 1. 首次同步推理

启动后先：

1. 连接机器人
2. 可选 `reset`
3. `ping` GR00T server
4. 读取第一帧 observation
5. 同步调用一次 `get_action()`

第一段动作会被截取为：

- `current_actions = actions[:open_loop_horizon]`

## 2. 逐步执行当前 chunk

当前 chunk 按索引 `action_idx` 一步一步执行。

每步会：

1. 如果到了预取触发点，就异步拉下一段 chunk
2. 发送当前动作到机器人
3. 用 `control_fps` 补齐节拍

## 3. 预取触发点

预取触发点按下面公式确定：

- `prefetch_start_idx = action_horizon - overlap - 1`

也就是在旧 chunk 还剩 `overlap` 步时，后台线程开始推理下一段动作。

异步预取实现位于：

- [\_start_async_prefetch()]( /home/f/Isaac-GR00T/robot_control_pc/core.py#L342 )

后台线程会：

1. 读取当前观测的 policy 格式
2. 对 state 做轻量 future-state-aware 前滚
3. 调用新的 `PolicyClient.get_action()`
4. 把结果暂存到 `_prefetch_actions`

## 4. Future-State-Aware 前滚

为了减少推理时延带来的状态错位，当前实现没有直接用“当前 observation”去推下一段，而是用 [\_project_future_state_observation()]( /home/f/Isaac-GR00T/robot_control_pc/core.py#L311 ) 对 state 做了一个近似前滚：

1. 复制当前 `policy_observation["state"]`
2. 取尚未执行的尾部动作 `remaining_actions`
3. 用其中最后一条动作覆盖 state 的最后一个时间步

当前这一步是轻量近似，主要对下列键有效：

- `left_arm` / `arm_left`
- `right_arm` / `arm_right`
- `left_gripper`
- `right_gripper`

## 5. 切换点

切换点按下面公式确定：

- `swap_idx = action_horizon - frozen - 1`

当执行到这里时，主线程会：

1. 等待后台异步结果完成
2. 取出下一段 chunk
3. 对当前 chunk 和下一段 chunk 做融合
4. 切换到新 chunk

这和你给的伪代码是对齐的：

```python
if i == action_horizon - overlap - 1:
    future = async policy.infer(new_obs)

robot.execute(actions[i])

if i == action_horizon - frozen - 1:
    actions = future.get()
    break
```

## 6. 融合逻辑

融合逻辑在 [\_fuse_action_chunks()]( /home/f/Isaac-GR00T/robot_control_pc/core.py#L253 )。

它做的事情是：

1. 取旧 chunk 的尾部 `overlap` 步
2. 取新 chunk 的前部 `overlap` 步
3. 在重叠区中：
   - 最后 `frozen` 步直接保留旧 chunk 的尾部
   - 前面的 `overlap - frozen` 步按线性权重混合

混合后的新 chunk 会再跳过：

- `start_offset = overlap - frozen`

这样就能让切换发生在正确的时间位置，不会把已经执行过的重叠段再执行一遍。

## 7. 控制节拍

节拍控制仍然是软实时方式：

1. 记录本步开始时间
2. 下发动作
3. 计算本步耗时
4. 若未达到 `1 / control_fps`，则 `sleep`

因此它仍然是：

- 异步推理 + 软实时控制循环

而不是硬实时 RTC 调度器。

## 8. 当前实现和 VLASH 的关系

当前版本借用了 VLASH 最重要的两层想法：

- 后台异步推理
- chunk 边界的 RTC 式重叠处理

但它仍然不是完整 VLASH，主要差别在：

- 没有完整的模型内 inpainting 机制
- 软融合是在 client 侧手工完成的
- future-state-aware 只做了轻量前滚
- 没有 action quantization
- 没有严格的延迟建模和 deadline 控制

## 9. 适合怎么调

如果动作边界还不够顺：

- 增大 `rtc_overlap`
- 适当减小 `rtc_frozen`

如果异步结果总是来不及：

- 增大 `rtc_frozen`
- 降低 `control_fps`
- 降低 server 推理延迟

如果动作太慢或拖：

- 减小 `point_time_from_start`
- 在保证稳定的前提下提高 `control_fps`

## 10. 一句话总结

当前 `robot_control_pc` 的执行逻辑是：

- 首段同步推理
- 中段异步预取下一段
- 在 chunk 尾部用 `overlap/frozen` 做 RTC 风格切换
- 用软融合减少新旧 chunk 的边界突变
