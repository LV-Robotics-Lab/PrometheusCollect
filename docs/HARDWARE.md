# 设备配置

## YAM

`robot=yam` 使用 `can_leader_l`、`can_leader_r`、`can_follower_l`、`can_follower_r`。
`robot=yam_drag` 只打开两个从臂 CAN。按现场接口配置这些名称；I2RT 的设备安装说明随
准备好的 SDK 源码提供。本项目不会自动修改主机 CAN 配置。

相机序列号位于各 robot 配置的 `cameras.serial_no`，逻辑位置为 `base_0`、
`left_wrist_0`、`right_wrist_0`。YAM 默认采集三路 640×480、30 Hz RGB。

## Sayo 踏板

在使用同款 Sayo CM6K 的主机上安装随项目提供的规则：

```bash
sudo install -m 0644 config/input/99-prometheus-sayo-pedal.rules /etc/udev/rules.d/
sudo usermod -aG plugdev "$USER"
sudo udevadm control --reload-rules
```

重新登录并重新插拔踏板后，检查 `/dev/input/prometheus-pedal` 是否可读。
使用 PCsensor 型号时选择相应规则文件，并按其键码核对 `event.pedal.pedal_map`。
不接踏板的采集命令添加 `event.input_source=tty`。

## 可选 ARX5 / Xense

保留 `robot=x5_1` 和 `robot=x5_2` 数采配置及 ARX5 夹爪标定工具。
这些配置包含 Xense 触觉设备，需安装供应商的 `xensesdk` 及所需运行库。
YAM 不加载或安装这些依赖。

ARX5 SDK 独立获取，不作为本项目训练子模块捆绑：

```bash
git clone https://github.com/MrSecant/arx5-sdk.git third_party/arx5-sdk
git -C third_party/arx5-sdk checkout ce0d1e76a9237de30908ab259dd9f3e4621056cf
```

按 SDK 说明准备编译依赖与兼容的 SOEM 1.4.0，然后运行
`PYTHON_BIN=/path/to/arx-python bash prometheus/tools/install_local_arx5_sdk.sh`。
安装脚本默认使用 `third_party/arx5-sdk`，也接受 `ARX5_SDK_ROOT` 指定外部 SDK。
它会检查 SOEM 库和构建结果，再把 Python 扩展安装到指定环境。

```bash
bash scripts/start_data_collection.sh robot=x5_1 user.task_id=my_task
```

ARX5 / Xense 的本次验证范围为配置和软件接口测试，未执行设备运动或触觉采集。
