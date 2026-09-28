# 板端构建、检查与验收

## 依赖与工具链

首选直接使用 `install/cross` 中的同一组库。目录包括 `bin/person_detect`、`lib/`、`model/best.rknn`。源码默认依赖 `third_party/opencv`，它是 AArch64 构建，不能用于 x86 主机集成测试。板端系统需要提供 glibc、libstdc++、libgcc_s、pthread 和 libm。

Debian/Ubuntu 板端如需使用系统 OpenCV 原生编译，可安装：

```bash
sudo apt-get update
sudo apt-get install build-essential cmake pkg-config libopencv-dev \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev python3

# 指定系统 OpenCV，避免选中附带库
OpenCV_DIR=/usr/lib/aarch64-linux-gnu/cmake/opencv4 bash scripts/build.sh native
```

系统路径随发行版而异，可用 `find /usr -name OpenCVConfig.cmake` 定位。OpenCV 必须有 `core/imgproc/imgcodecs/videoio`；没有 videoio 时 CMake 配置失败，没有 FFmpeg 时视频工厂报错。编译选项与依赖说明参见 [OpenCV 官方配置文档](https://docs.opencv.org/4.13.0/db/d05/tutorial_config_reference.html)。

交叉工具链可覆盖 `CROSS_COMPILE`，例如现有 GCC 10.3 工具链：

```bash
CROSS_COMPILE=/home/leon/opt/tool_chain/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu- \
bash scripts/build.sh cross
```

更换工具链时使用新的 `BUILD_DIR`，避免 CMake 缓存旧编译器。附带库使用通用 AArch64/NEON，没有 RK3588 核选择或高端 CPU 专属指令。

## 环境检查

```bash
bash scripts/check_board.sh /opt/person_detector
ldd /opt/person_detector/bin/person_detect
readelf -d /opt/person_detector/bin/person_detect
strings /usr/lib/aarch64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4.26
getconf GNU_LIBC_VERSION

# 路径随内核变化；debugfs 可能需要 root 读取
cat /sys/kernel/debug/rknpu/version
cat /sys/kernel/debug/rknpu/load
cat /sys/kernel/debug/rga/driver_version
dmesg | grep -Ei 'rknpu|rga'
```

程序启动会输出 Runtime/驱动版本，打开视频时输出 FFmpeg 后端及 FPS。RKNN 头文件与 Runtime 均来自本地 Model Zoo 的 2.3.2 版本，模型也由 2.3.2 构建。遇到 Runtime/驱动不匹配时使用板厂兼容 SDK；可通过 `RKNN_ROOT` 替换，不需要重新转换模型来解决应用的库路径问题。

RGA 头文件和库为同一套 1.10.0。本项目检查 RGA 返回值和 stride，但主机链接无法验证板端 RGA2 驱动、DMA 地址可达性或缓存行为。先用普通路径建立正确基线，再显式运行 `--io bound --preprocess opencv`、`--io bound --preprocess rga`，失败日志直接指出阶段；自动路径的实际选择记录在 JSONL 和 summary 中。

## 正确性验收顺序

1. 使用空画面、多行人、横图、竖图、奇数尺寸图片，验证 `--io standard --preprocess opencv --workers 1`。
2. 用 `--dump-first` 导出同一帧九个原始输出，运行 NumPy/C++ 参考比较；同时核对 `input.rgb` 黑边和 RGB 通道。
3. 用相同输入比较绑定/OpenCV 与普通/OpenCV。接着比较绑定/RGA；可以先按 IoU/分数容差比较，再人工检查阈值附近结果。不要因 FPS 更高就忽略检测差异。
4. 对短视频和完整视频检查 JSONL 帧号连续、帧数等于实际解码帧数；标注视频重新解码核对输出帧数。
5. 分别使用 1/2/3/4 workers、较小 `--inflight` 和标注输出制造回压。中断或推理失败必须返回非零，线程全部退出。
6. 使用代表性长视频持续运行，观察 RSS、NPU 负载、温度、频率和输出计数。固定帧槽限制图像内存，但帧尺寸改变会调整槽内图像缓冲。

未接入摄像头、RTSP、实时丢帧及 MPP。当前读取路径面向本地有限文件，取消会等待正在进行的 RKNN、RGA、解码或编码调用结束，无法在应用层抢占卡在驱动中的调用。

## 测速条件

同一模型、同一输入、同一阈值、相同电源/散热/频率条件，每组至少重复三次。先跑无标注，再跑有标注；用于部署推荐的数据应来自真实 RK3568，不能使用主机模拟 Runtime 的统计。`benchmark.py` 保存实际工作模式和逐次数据，故自动回退不会被误记为成功使用 RGA。

首版使用 OpenCV/FFmpeg 软件解码和可选软件编码。若性能摘要显示 read/output 是主要瓶颈，可在后续增加 MPP 输入或输出工厂；如果 inference 是主要瓶颈，先比较 worker 数，RK3568 单核 NPU 不保证四个 worker 比一个更快。
