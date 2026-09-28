#!/usr/bin/env python3
"""Run the correctness-gated 1/2/3/4 worker matrix on a real RK3568 board."""
import argparse
import hashlib
import json
import pathlib
import platform
import statistics
import subprocess
import time
from compare_jsonl import compare

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', type=pathlib.Path, required=True)
parser.add_argument('--model', type=pathlib.Path, required=True)
parser.add_argument('--input', type=pathlib.Path, required=True)
parser.add_argument('--out', type=pathlib.Path, default=pathlib.Path('benchmark_results'))
parser.add_argument('--repeat', type=int, default=3)
parser.add_argument('--warmup', type=int, default=5)
parser.add_argument('--annotated', choices=['none', 'image', 'video'], default='none', help='Run both unannotated and annotated matrices')
parser.add_argument('--timeout', type=float, default=3600)
parser.add_argument('--conf', type=float, default=.25)
parser.add_argument('--nms', type=float, default=.45)
parser.add_argument('--iou', type=float, default=.98)
parser.add_argument('--score-tolerance', type=float, default=.02)
args = parser.parse_args()
if platform.machine().lower() not in ('aarch64', 'arm64'):
    parser.error('Performance recommendations require running on the RK3568 AArch64 board')
if args.repeat < 1:
    parser.error('--repeat must be positive')
args.out.mkdir(parents=True, exist_ok=True)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


inputs = sorted(p for p in args.input.iterdir() if p.is_file()) if args.input.is_dir() else [args.input]
manifest = {'model_sha256': digest(args.model), 'input_sha256': {str(p): digest(p) for p in inputs},
            'platform': platform.platform(), 'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'arguments': vars(args).copy()}
manifest['arguments'] = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in manifest['arguments'].items()}
(args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
groups = ['none'] if args.annotated == 'none' else ['none', args.annotated]
results, recommendations = [], {}
for annotation in groups:
    baseline = None
    group = []
    for preprocessing, io in [('opencv', 'standard'), ('opencv', 'bound'), ('rga', 'bound'), ('auto', 'auto')]:
        for workers in (1, 2, 3, 4):
            label = f'{annotation}_{preprocessing}_{io}_w{workers}'
            samples, failure = [], None
            for repeat in range(args.repeat):
                name = f'{label}_r{repeat}'
                jsonl, summary = args.out / (name + '.jsonl'), args.out / (name + '.json')
                command = [str(args.binary.resolve()), '--input', str(args.input.resolve()), '--model', str(args.model.resolve()),
                           '--workers', str(workers), '--preprocess', preprocessing, '--io', io,
                           '--conf', str(args.conf), '--nms', str(args.nms), '--warmup', str(args.warmup),
                           '--jsonl', str(jsonl), '--summary', str(summary)]
                if annotation != 'none':
                    command += ['--annotated', str(args.out / (name + '.mp4' if annotation == 'video' else name + '_images'))]
                try:
                    with open(args.out / (name + '.log'), 'w') as log:
                        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                    if process.returncode:
                        raise RuntimeError(f'Exit {process.returncode}; see {name}.log')
                    if 'MOCK-TEST-ONLY' in (args.out / (name + '.log')).read_text():
                        raise RuntimeError('Mock Runtime cannot produce board performance recommendations')
                    stats = json.loads(summary.read_text())
                    if baseline is None:
                        if (preprocessing, io, workers) != ('opencv', 'standard', 1):
                            raise RuntimeError('Single-worker standard/OpenCV baseline failed; no valid comparison baseline')
                        baseline = jsonl
                    actual_frames = compare(baseline, jsonl, args.iou, args.score_tolerance)
                    if actual_frames != stats['frames'] or stats['submitted'] != stats['frames']:
                        raise ValueError('Summary/frame count mismatch')
                    if actual_frames == 0:
                        raise ValueError('Zero-frame input is not a performance benchmark')
                    if any(stats[k] > stats['capacity'] for k in ('inflight_peak', 'task_queue_peak', 'result_queue_peak')):
                        raise ValueError('Frame capacity exceeded')
                    samples.append(stats)
                except (RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                    failure = str(error)
                    break
            result = {'configuration': label, 'correct': failure is None, 'failure': failure, 'samples': samples}
            if failure is None:
                result['median_end_to_end_fps'] = statistics.median(s['end_to_end_fps'] for s in samples)
                result['median_detection_fps'] = statistics.median(s['detection_fps'] for s in samples)
            results.append(result)
            group.append(result)
            print(label, failure or f"{result['median_end_to_end_fps']:.3f} end-to-end FPS", flush=True)
    if group[0]['correct']:
        best = max((r for r in group if r['correct']), key=lambda r: r['median_end_to_end_fps'])
        # Require a measurable margin; all successful repetitions already passed result comparison.
        recommended = best if best['median_end_to_end_fps'] > group[0]['median_end_to_end_fps'] * 1.02 else group[0]
        recommendations[annotation] = recommended['configuration']
(args.out / 'results.json').write_text(json.dumps({'results': results, 'recommendations': recommendations}, indent=2))
print('Board recommendations:', recommendations)
