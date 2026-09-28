# 已完成验证与板端待验项目

记录日期：2026-09-28。主机为 x86_64，编译器 GCC 9.4、CMake 3.16；Python 使用 `/home/leon/miniconda3/envs/rknn/bin/python`。AArch64 交叉编译器为 `aarch64-linux-gnu-g++ 9.4.0`。

| 验证项 | 结果 | 验证范围 |
|---|---|---|
| Release 核心测试 | 2/2 通过 | 后处理、几何、JSON 转义、并发队列与取消 |
| NumPy/C++ 对照 | 通过 | 六种尺寸、三个网格尺度、随机 INT8 张量，同一输入张量解码 |
| 主机完整集成测试 | 通过 | 真实 OpenCV 图片/视频 I/O，模拟 RKNN API；不验证 NPU 数值 |
| AArch64 完整编译链接 | 通过 | 真实 RKNN 2.3.2、RGA 1.10.0、OpenCV 4.10.0、FFmpeg 4.4.5 |
| 安装包检查 | 通过 | AArch64 ELF、SONAME 依赖闭包、相对 RPATH、原模型 SHA256 |
| AddressSanitizer / UBSan | 通过 | 独立核心和线程池测试；关闭 LeakSanitizer |
| Python/Bash 语法 | 通过 | 部署、依赖构建、参考比较、性能工具 |

后处理测试包括空检测、多行人、阈值边界、超出 INT8 表示范围的阈值、极端 DFL logits、重叠框 NMS，以及横/竖/奇数尺寸的还原和裁剪。参考比较覆盖 416×416、1920×1080、1080×1920、641×479、479×641、832×421；坐标绝对误差容限为 0.002 像素。

并发测试覆盖 0/1/2/19/101 帧、四 worker 乱序完成、慢输出、队列满、单帧槽、EOF、读取/检测/输出失败和取消。成功时帧号连续、输出计数等于提交计数、所有队列峰值不超过总帧槽容量。

集成测试处理 12 张混合尺寸图片；比较 1/3/4 workers 和 standard/bound/auto 的完整检测输出。模拟 native 输入设置 `w_stride=448`，验证 strided UINT8 NHWC 绑定路径。检查 `input.rgb` 与 Python 黑边/BGR→RGB 参考，跨 OpenCV 版本像素误差不超过 1。17 帧 MJPEG 输入视频生成 17 条 JSONL 和可重新解码的 17 帧 MPEG-4 标注视频。

失败场景包括错误模型输出数、绑定查询失败、绑定首轮推理失败、显式模式不支持、运行中推理失败、`/dev/full` 写入失败、SIGINT 和输出路径覆盖输入的拒绝。所有场景都在测试超时范围内返回非零。

主机沙箱使 LeakSanitizer 报告 ptrace 不兼容，因此 AddressSanitizer/UBSan 测试使用 `ASAN_OPTIONS=detect_leaks=0`，不把它记为泄漏检测通过。可以在普通主机或板端另行运行泄漏检查。

## 可复现命令

```bash
bash scripts/host_tests.sh
HOST_OPENCV_DIR=/path/to/host/opencv/lib/cmake/opencv4 bash scripts/host_tests.sh
bash scripts/build.sh cross
python3 scripts/check_package.py install/cross
```

主机集成测试所用 host OpenCV 位于本次独立构建的 `/tmp/person-rk3568-deps/opencv-host/lib/cmake/opencv4`；如果临时目录已清理，使用 `bash scripts/build_opencv.sh host` 重建。依赖重建不会改变 `rknn` conda 环境或系统库。

## 必须在 RK3568 完成的验收

- 真实模型输出与现有 Python 验证结果比较，包括空画面、多行人及阈值附近样本。
- 普通/绑定输入检测一致性，RGA/OpenCV 结果差异，以及实际 stride、DMA FD、缓存同步与驱动兼容性。
- 1/2/3/4 worker 测速，无标注和有标注分别比较。只推荐正确性通过且实测更快的配置。
- 长视频持续运行的 RSS、温度、频率、NPU 负载、帧数和退出行为。

当前未连接 RK3568，因此没有执行 NPU 推理、板端持续运行或性能测量。`install/cross` 已通过编译链接与包检查，仍需上述硬件验收；本记录不提供模拟 FPS 作为部署性能。
