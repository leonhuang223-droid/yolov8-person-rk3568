# 依赖来源

- `rknn/include/rknn_api.h`、`rknn/lib/librknnrt.so`：原工作区 `rknn_model_zoo-main/3rdparty/rknpu2`，Runtime 2.3.2。头文件保留 Rockchip 原始声明。
- `rga/include`、`rga/lib/librga.so`：原工作区 `rknn_model_zoo-main/3rdparty/librga`，API 1.10.0。保留各头文件原始版权与许可声明。
- `opencv/`：OpenCV 4.10.0 官方源码 AArch64 Release 构建，core/imgproc/imgcodecs/videoio，链接 FFmpeg 4.4.5 的静态库；OpenCV 动态库 RPATH 为 `$ORIGIN`。FFmpeg 未启用 GPL/nonfree 组件。许可文本在 `licenses/`。
- FFmpeg 4.4.5 源码：[官方发行源码](https://ffmpeg.org/releases/ffmpeg-4.4.5.tar.xz)。OpenCV 4.10.0 源码：[官方仓库](https://github.com/opencv/opencv/tree/4.10.0)。可运行 `scripts/build_opencv.sh` 重建并替换附带二进制。
- 模型复制自已有 RK3568 `best.rknn`，SHA256 及接口信息见 `model/metadata.json`，未训练或转换。

项目业务源码为本次新建；依赖和模型维持其原始版权声明及适用条件。
