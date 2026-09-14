# 拆分验证记录

日期：2026-09-14。项目：`prometheus-collect 0.1.0`。

## 拆分范围

从包含上一轮 YAM 修复的工作区提取数采代码，形成独立项目。
原始文件校验值和基线提交记录在 `SOURCE_SNAPSHOT.json`，原项目未因本次拆分而删除文件。

保留：YAM 主从臂、YAM 从臂拖拽、ARX5、RealSense、Xense、踏板/键盘事件、
硬件健康检查、异步原始录制、数据读取、曲线图、预览和可选上传。

移除：训练项目、策略及调度器、DAgger、rollout、推理环境和客户端、ZMQ 模型服务，
以及 Flow Matching、HumanEgo、FoundationStereo 子模块、相关配置、脚本和测试。
没有复制旧数据集、模型权重、旧虚拟环境、构建产物或 Git 历史。

具体解耦：

- 采集预览不再导入 DAgger，也不保留其状态界面。
- 测试中的通用录制载荷与无硬件客户端不再依赖策略模块。
- 默认采集配置为 YAM，数据根目录和 Home 路径不再绑定原项目位置。
- 上传不再预选某个用户账号或已采集的数据目录，默认仍为本地 dry run。
- I2RT 准备脚本可独立获取锁定提交，也可从本地 Git 仓库离线导出。
- wheel 包含 Hydra 配置和三个命令行入口，源码包额外包含安装脚本、设备规则和测试。

## 测试结果

| 检查 | 结果 |
| --- | --- |
| 独立 Python 3.12 环境的数采回归 | **164 passed, 3 skipped** |
| 锁定 I2RT 的关闭、拖拽调参和运行健康测试 | **26 passed** |
| Ruff 静态检查 | 通过 |
| Python 编译、启动 shell 语法检查 | 通过 |
| YAM 两种模式的 SDK API 预检 | 通过，未打开 CAN |
| 安装后 SDK 的 YAM Ultra 2 + linear_4310 模型加载 | 通过，`nq=8, nv=8`，未打开 CAN |
| 独立基础环境安装 wheel | 通过 |
| 源码目录之外导入 wheel 全部模块 | 通过 |
| 安装后解析 YAM、YAM drag、X5 1、X5 2 配置 | 通过 |
| 安装后无硬件采集、事件保存与清单检查 | 通过 |
| 安装后上传本地 dry run、episode 帮助入口 | 通过，未联网或上传 |
| wheel / 源码包内容检查 | 配置、脚本齐全，无训练模块、缓存或数据产物 |

三个跳过项分别是 ARX5、RealSense 和踏板的真实硬件测试，需显式开启相应环境变量。
完整数采测试使用项目新建的 `.venv` 和安装后的 I2RT 包；没有引用原 Prometheus 的 Python 包。
wheel 基础环境 `.venv-wheel-check` 不含 ROS、I2RT、PyTorch 或 Transformers，仍完成了全部
模块导入、配置解析和无硬件采集。这证明基础数采包可以独立安装，实际设备运行另需其 SDK。

新环境已验证的主要版本：Python 3.12.3、NumPy 1.26.4、Hydra 1.3.6、
OpenCV headless 4.11.0.86、Matplotlib 3.11.1、I2RT 1.1.2（锁定提交与兼容补丁）、
MuJoCo 3.12.0、SciPy 1.16.3。

本地详细输出位于 `outputs/software_tests.txt`、`outputs/i2rt_tests.txt` 和
`outputs/distribution_check.txt`。这些运行输出不打进源码包。

## 继承的 YAM 修复与验证边界

YAM 驱动、源时间戳传递、录制器及硬件健康检查关键文件与原修复版本逐字节一致。
保留了反馈过期/线程失效检测、Home 与示教的互斥、双臂独立恢复、单侧动作记录、
保存失败标记、清理顺序和 YAM 无 EEF 时的完整性判定等修复。

本次拆分没有更改机械参数、夹爪助力方向、归位速度或动作限幅。
没有进行真实机械臂运动；Home 返回、拖拽助力与松手表现、相机长时间丢帧情况仍需现场验收。

## 交付

`dist/prometheus_collect-0.1.0.tar.gz` 是可分发的源码项目，包含中文 README、
配置、SDK 准备脚本、设备规则与测试。`dist/prometheus_collect-0.1.0-py3-none-any.whl`
是可安装的采集包。实际 YAM 部署按 README 准备 I2RT 和 ROS。
本地虚拟环境用于验证和当前机器运行，跨机器部署应重新创建环境。
