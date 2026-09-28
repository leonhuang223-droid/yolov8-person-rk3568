#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:-native}
build_dir=${BUILD_DIR:-"$project_dir/build/$mode"}
install_dir=${INSTALL_DIR:-"$project_dir/install/$mode"}
args=(-S "$project_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$install_dir")
case "$mode" in
  host-tests) args+=(-DPERSON_BUILD_APP=OFF) ;;
  cross) args+=(-DCMAKE_TOOLCHAIN_FILE="$project_dir/cmake/aarch64.cmake" -DPERSON_BUILD_TESTS=OFF) ;;
  native) args+=(-DPERSON_BUILD_TESTS=OFF) ;;
  *) echo 'Usage: build.sh native|cross|host-tests [extra CMake arguments]' >&2; exit 2 ;;
esac
if [[ -n ${OpenCV_DIR:-} ]]; then args+=(-DOpenCV_DIR="$OpenCV_DIR"); fi
if [[ -n ${RKNN_ROOT:-} ]]; then args+=(-DRKNN_ROOT="$RKNN_ROOT"); fi
if [[ -n ${RGA_ROOT:-} ]]; then args+=(-DRGA_ROOT="$RGA_ROOT"); fi
if [[ -n ${PERSON_ENABLE_RGA:-} ]]; then args+=(-DPERSON_ENABLE_RGA="$PERSON_ENABLE_RGA"); fi
shift "$(( $# > 0 ? 1 : 0 ))"
cmake "${args[@]}" "$@"
cmake --build "$build_dir" --parallel "${JOBS:-2}"
if [[ "$mode" == host-tests ]]; then
  (cd "$build_dir" && ctest --output-on-failure)
else
  cmake --install "$build_dir"
fi
