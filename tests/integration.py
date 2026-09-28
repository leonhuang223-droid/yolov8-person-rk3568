#!/usr/bin/env python3
"""Exercise real OpenCV I/O, factories, stride, ordered output and failures with a mock RKNN API."""
import argparse
import json
import os
import pathlib
import signal
import subprocess
import tempfile
import time
import cv2
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', type=pathlib.Path, required=True)
parser.add_argument('--model', type=pathlib.Path, required=True)
args = parser.parse_args()
binary, model = args.binary.resolve(), args.model.resolve()


def validate(path, frames):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r['frame_id'] for r in rows] == list(range(frames))
    return rows


with tempfile.TemporaryDirectory() as temp:
    root = pathlib.Path(temp)
    images = root / 'images'
    images.mkdir()
    for i in range(12):
        width, height = [(641, 479), (479, 641), (832, 421)][i % 3]
        rng = np.random.default_rng(i)
        cv2.imwrite(str(images / f'{i:03}.png'), rng.integers(0, 256, (height, width, 3), dtype=np.uint8))

    def run(name, input_path=images, extra=(), env=None, success=True):
        jsonl, summary = root / (name + '.jsonl'), root / (name + '.json')
        command = [str(binary), '--input', str(input_path), '--model', str(model), '--jsonl', str(jsonl),
                   '--summary', str(summary), '--warmup', '0', '--preprocess', 'opencv'] + list(extra)
        process = subprocess.run(command, capture_output=True, text=True, timeout=20,
                                 env={**os.environ, **(env or {})})
        assert (process.returncode == 0) == success, (command, process.stdout, process.stderr)
        return jsonl, summary, process

    baseline, _, _ = run('baseline', extra=['--io', 'standard', '--workers', '1', '--dump-first', str(root / 'dump')])
    rows = validate(baseline, 12)
    for io, workers in [('bound', 4), ('auto', 3), ('standard', 4)]:
        path, summary, _ = run(io, extra=['--io', io, '--workers', str(workers), '--inflight', '3'])
        actual = validate(path, 12)
        assert [r['detections'] for r in actual] == [r['detections'] for r in rows]
        stats = json.loads(summary.read_text())
        assert stats['capacity'] == 3 and stats['inflight_peak'] <= 3
    image = cv2.imread(str(images / '000.png'))
    scale = min(416 / image.shape[1], 416 / image.shape[0])
    new_w, new_h = round(image.shape[1] * scale), round(image.shape[0] * scale)
    expected = np.zeros((416, 416, 3), dtype=np.uint8)
    left, top = (416 - new_w) // 2, (416 - new_h) // 2
    expected[top:top + new_h, left:left + new_w] = cv2.cvtColor(cv2.resize(image, (new_w, new_h)), cv2.COLOR_BGR2RGB)
    actual = np.fromfile(root / 'dump' / 'input.rgb', dtype=np.uint8).reshape(416, 416, 3)
    assert np.abs(actual.astype(int) - expected.astype(int)).max() <= 1
    path, _, process = run('fallback', env={'MOCK_NO_BOUND': '1'})
    assert 'use standard' in process.stderr and all(r['io'] == 'standard' for r in validate(path, 12))
    path, _, process = run('bound_probe_fallback', env={'MOCK_BOUND_FAIL': '1'})
    assert 'use standard' in process.stderr and all(r['io'] == 'standard' for r in validate(path, 12))
    run('explicit_bound_failure', extra=['--io', 'bound'], env={'MOCK_NO_BOUND': '1'}, success=False)
    run('wrong_model', env={'MOCK_BAD_MODEL': '1'}, success=False)
    run('inference_failure', env={'MOCK_FAIL_RUN': '4'}, extra=['--io', 'standard', '--workers', '4'], success=False)
    run('missing_rga', extra=['--preprocess', 'rga'], success=False)
    # Input must survive a rejected output alias.
    source_bytes = (images / '000.png').read_bytes()
    command = [str(binary), '--input', str(images / '000.png'), '--jsonl', str(images / '000.png')]
    process = subprocess.run(command, capture_output=True, timeout=5)
    assert process.returncode and (images / '000.png').read_bytes() == source_bytes
    # A disk write failure must cancel all waiting workers.
    if pathlib.Path('/dev/full').exists():
        run('output_failure', extra=['--jsonl', '/dev/full', '--workers', '4'], success=False)
    # A single image and annotations use the same ordered output path.
    path, _, _ = run('single', images / '000.png', extra=['--annotated', str(root / 'marked')])
    validate(path, 1)
    assert len(list((root / 'marked').glob('*.jpg'))) == 1
    video = root / 'input.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 12, (320, 240))
    assert writer.isOpened()
    for i in range(17):
        writer.write(np.full((240, 320, 3), i * 10, dtype=np.uint8))
    writer.release()
    path, _, _ = run('video', video, extra=['--workers', '4', '--annotated', str(root / 'marked.mp4')])
    validate(path, 17)
    capture = cv2.VideoCapture(str(root / 'marked.mp4'))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    assert decoded == 17
    # Every thread inherits blocked termination signals; sigwait cancels full queues safely.
    command = [str(binary), '--input', str(images), '--model', str(model), '--jsonl', str(root / 'cancel.jsonl'),
               '--io', 'standard', '--preprocess', 'opencv', '--workers', '2', '--inflight', '2', '--warmup', '0']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env={**os.environ, 'MOCK_SLEEP_MS': '200'})
    time.sleep(.1)
    process.send_signal(signal.SIGINT)
    stdout, stderr = process.communicate(timeout=5)
    assert process.returncode > 0 and b'Interrupted by signal' in stderr
print('Host integration passed: images/video, bound stride, factories, ordered counts, annotation, fallbacks/errors, SIGINT')
