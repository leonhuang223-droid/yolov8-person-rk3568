#!/usr/bin/env bash
# Standalone pinned dependencies. No sudo, conda changes, or system library replacement.
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:-cross}
deps_dir=${DEPS_DIR:-"$project_dir/build/deps-$mode"}
prefix=${DEPS_PREFIX:-"$project_dir/third_party/opencv"}
mkdir -p "$deps_dir"
deps_dir=$(cd -- "$deps_dir" && pwd)
mkdir -p "$prefix"
prefix=$(cd -- "$prefix" && pwd)
python3 - "$deps_dir" <<'PY'
import pathlib, sys, tarfile, urllib.request
root=pathlib.Path(sys.argv[1])
packages=[('opencv-4.10.0','https://codeload.github.com/opencv/opencv/tar.gz/refs/tags/4.10.0', '.tar.gz'),
          ('ffmpeg-4.4.5','https://ffmpeg.org/releases/ffmpeg-4.4.5.tar.xz', '.tar.xz')]
for name, url, suffix in packages:
    if (root/name).is_dir():
        continue
    archive=root/(name+suffix)
    if not archive.exists():
        temporary=archive.with_suffix(archive.suffix+'.download')
        urllib.request.urlretrieve(url,temporary)
        temporary.rename(archive)
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            resolved=(root/member.name).resolve()
            if root.resolve() not in resolved.parents:
                raise ValueError('Unsafe source archive path')
        source.extractall(root)
PY
ffmpeg_prefix="$deps_dir/ffmpeg-install"
ffmpeg_args=(--prefix="$ffmpeg_prefix" --disable-programs --disable-doc --disable-debug --disable-network --disable-everything
  --enable-pic --enable-avcodec --enable-avformat --enable-avutil --enable-swscale --enable-protocol=file
  --enable-demuxer=mov,avi,image2 --enable-decoder=h264,mpeg4,mjpeg --enable-encoder=mpeg4,mjpeg
  --enable-muxer=mp4,avi --enable-parser=h264,mpeg4video)
cmake_args=()
case "$mode" in
  cross)
    ffmpeg_args+=(--enable-cross-compile --cross-prefix="${CROSS_COMPILE:-aarch64-linux-gnu-}" --arch=aarch64 --target-os=linux)
    cmake_args+=(-DCMAKE_TOOLCHAIN_FILE="$project_dir/cmake/aarch64.cmake") ;;
  native) ;;
  host)
    ffmpeg_args+=(--disable-asm)
    if [[ -z ${DEPS_PREFIX:-} ]]; then prefix="$deps_dir/opencv-host"; fi ;;
  *) echo 'Usage: build_opencv.sh cross|native|host' >&2; exit 2 ;;
esac
mkdir -p "$deps_dir/ffmpeg-build"
(cd "$deps_dir/ffmpeg-build"
 "$deps_dir/ffmpeg-4.4.5/configure" "${ffmpeg_args[@]}"
 make -j"${JOBS:-2}"
 make install)
PKG_CONFIG_LIBDIR="$ffmpeg_prefix/lib/pkgconfig" cmake -S "$deps_dir/opencv-4.10.0" -B "$deps_dir/opencv-build" "${cmake_args[@]}" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$prefix" '-DCMAKE_INSTALL_RPATH=$ORIGIN' \
  -DBUILD_LIST=core,imgproc,imgcodecs,videoio -DWITH_FFMPEG=ON -DOPENCV_ENABLE_PKG_CONFIG=ON \
  -DWITH_GSTREAMER=OFF -DWITH_V4L=OFF -DWITH_OPENCL=OFF -DWITH_IPP=OFF -DWITH_LAPACK=OFF \
  -DWITH_TIFF=OFF -DWITH_OPENEXR=OFF -DWITH_WEBP=OFF -DWITH_ITT=OFF -DWITH_OBSENSOR=OFF \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF \
  -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_SHARED_LINKER_FLAGS="-L$ffmpeg_prefix/lib" -DCMAKE_EXE_LINKER_FLAGS="-L$ffmpeg_prefix/lib"
if ! python3 - "$deps_dir/opencv-build/opencv2/cvconfig.h" <<'PY'
import pathlib, sys
sys.exit(0 if '#define HAVE_FFMPEG' in pathlib.Path(sys.argv[1]).read_text() else 1)
PY
then
  echo 'OpenCV did not enable FFmpeg; check pkg-config and linker output above.' >&2; exit 1
fi
cmake --build "$deps_dir/opencv-build" --parallel "${JOBS:-2}"
cmake --install "$deps_dir/opencv-build"
echo "OpenCV_DIR=$prefix/lib/cmake/opencv4"
