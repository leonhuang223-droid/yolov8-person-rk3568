#!/usr/bin/env python3
"""Verify packaged ELF architecture, SONAME closure, relative RPATH and model checksum without running ARM binaries."""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('package', type=pathlib.Path)
args = parser.parse_args()
binary = args.package / 'bin' / 'person_detect'
libs = args.package / 'lib'
system = {'libc.so.6', 'libm.so.6', 'libpthread.so.0', 'libstdc++.so.6', 'libgcc_s.so.1',
          'libdl.so.2', 'librt.so.1', 'libatomic.so.1', 'ld-linux-aarch64.so.1'}
for path in [binary] + sorted(p for p in libs.iterdir() if p.is_file() and not p.is_symlink()):
    header = subprocess.check_output(['readelf', '-h', str(path)], text=True)
    if 'AArch64' not in header:
        raise ValueError(f'Wrong architecture: {path}')
    dynamic = subprocess.check_output(['readelf', '-d', str(path)], text=True)
    for needed in re.findall(r'NEEDED.*?\[(.*?)\]', dynamic):
        if needed not in system and not (libs / needed).is_file():
            raise ValueError(f'Missing shared dependency: {path.name} -> {needed}')
    for search in re.findall(r'(?:RUNPATH|RPATH).*?\[(.*?)\]', dynamic):
        if any(part and not part.startswith('$ORIGIN') for part in search.split(':')):
            raise ValueError(f'Non-relative library search path in {path}: {search}')
if '$ORIGIN/../lib' not in subprocess.check_output(['readelf', '-d', str(binary)], text=True):
    raise ValueError('Executable lacks relative library RPATH')
metadata_path = pathlib.Path(__file__).resolve().parents[1] / 'model' / 'metadata.json'
expected = json.loads(metadata_path.read_text())['sha256']
actual = hashlib.sha256((args.package / 'model' / 'best.rknn').read_bytes()).hexdigest()
if actual != expected:
    raise ValueError('Packaged model differs from reviewed original')
print('Package passed: AArch64 ELF, shared dependencies, relative RPATH, original model SHA256')
