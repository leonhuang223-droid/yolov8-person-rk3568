# 图片与视频测试素材

`test.jpg` 原样复制自工作区 `rknn_model_zoo-main/examples/yolov8/model/bus.jpg`，画面包含公交车及多个行人。安装时复制到包根目录，供 `--input ./test.jpg` 直接使用。

用于检查读取、预处理、模型推理、JSONL 和标注输出链路；检测精度仍应使用你的代表性行人数据集评估。

`test.mp4` 来自网上下载的 [OpenCV 官方示例 vtest.avi](https://github.com/opencv/opencv/blob/4.10.0/samples/data/vtest.avi)，画面为固定摄像机下的多名移动行人。保留前 20 秒，转为部署包支持的 MPEG-4 MP4：768×576、10 FPS、200 帧，约 2.88 MiB。安装时复制到包根目录，可使用 `--input ./test.mp4 --annotated results/marked.mp4` 查看模型检测效果。

此文件为未加检测框的输入素材，检测标注由板端实际 RKNN 推理生成。成功完整处理后可执行 `python3 tools/check_jsonl.py results/video.jsonl --frames 200` 检查输出帧数与连续性。原始下载文件保存在源码目录的 `samples/vtest.avi`；来源链接、编码信息和 SHA256 在 `samples/video_metadata.json`。
