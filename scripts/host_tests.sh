#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PERSON_PYTHON:-/home/leon/miniconda3/envs/rknn/bin/python}
if [[ ! -x "$python_bin" ]]; then python_bin=$(command -v python3); fi
bash "$project_dir/scripts/build.sh" host-tests
"$python_bin" "$project_dir/scripts/reference_decode.py" --decoder "${BUILD_DIR:-$project_dir/build/host-tests}/decode_dump"
if [[ -n ${HOST_OPENCV_DIR:-} ]]; then
  cmake -S "$project_dir" -B "$project_dir/build/integration" -DCMAKE_BUILD_TYPE=Release \
    -DPERSON_USE_MOCK_RUNTIME=ON -DOpenCV_DIR="$HOST_OPENCV_DIR"
  cmake --build "$project_dir/build/integration" --parallel "${JOBS:-2}"
  "$python_bin" "$project_dir/tests/integration.py" --binary "$project_dir/build/integration/person_detect" --model "$project_dir/model/best.rknn"
fi
