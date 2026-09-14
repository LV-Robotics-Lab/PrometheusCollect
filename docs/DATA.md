# 原始采集数据

`HardwareSession` 声明设备流，`DataSession` 通过 ROS 订阅、添加接收时间并送入异步
录制队列。相机、机器人、动作和事件保留各自采样频率，不强行重采样到训练帧率。

```text
<task_id>/0000/
  camera/                  RGB MP4、可选 IR MKV、timestamp.pkl、metadata.json
  tactile/                 可选触觉视频、流和时间戳
  robot/robot_state_dict.pkl
  action/executed_action_dict.pkl
  event/event_dict.pkl
  manifests/               各数据模态的保存状态和计数
```

机器人状态按源时间组织左右臂的 joint、vel、effort 和 EEF。YAM 只声明 joint state，
EEF 留空且不计入缺失；ARX5 声明 EEF 时仍校验完整性。视频和 action 字段结构沿用
原始数采格式。YAM action 是左右各六关节加夹爪的 14 维 `abs_qpos`，夹爪单位为米。

主从采集记录实际跟随目标及同步状态；拖拽采集记录观测到的位置，便于后续作为示教
动作使用。两者并非同一种命令来源，处理数据时应保留其 metadata。

没有该类数据时，不要求存在对应目录。保存失败的 episode 在清单中标记为 failed，
不会作为成功保存计数；默认上传检查拒绝未完整保存的数据。

读入自己采集的 episode：

```python
from prometheus.data.episode import RawEpisodeReader

episode = RawEpisodeReader('/data/demonstrations/my_task/0000')
print(episode.counts())
```

PKL 用于可信的本地原始数据，不用于读取来源不明的文件。
读写层保留历史 `action_chunk` 字段的兼容能力，以便读取已有数据；不包含产生这些字段的
策略、调度器、推理服务或训练代码。
