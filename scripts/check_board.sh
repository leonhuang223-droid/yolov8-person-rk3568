#!/usr/bin/env bash
set -euo pipefail
script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -x "$script_root/bin/person_detect" ]]; then default_package="$script_root"; else default_package="$script_root/install/native"; fi
package_dir=${1:-"$default_package"}
uname -a
getconf LONG_BIT
if [[ -r /proc/device-tree/compatible ]]; then tr '\0' '\n' </proc/device-tree/compatible; fi
ldd "$package_dir/bin/person_detect"
readelf -d "$package_dir/bin/person_detect" | sed -n '/NEEDED\|RUNPATH\|RPATH/p'
for path in /sys/kernel/debug/rknpu/version /sys/kernel/debug/rknpu/load /sys/kernel/debug/rga/driver_version /sys/kernel/debug/rga/version; do
  if [[ -r "$path" ]]; then echo "$path"; cat "$path"; fi
done
if command -v opencv_version >/dev/null; then opencv_version --verbose; fi
"$package_dir/bin/person_detect" --help
echo 'Runtime/driver versions and actual FFmpeg backend are logged at model/video startup.'
