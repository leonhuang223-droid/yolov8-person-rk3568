#!/usr/bin/env python3
"""Independent NumPy decode; compare synthetic fixtures or board raw-output dumps."""
import argparse
import json
import pathlib
import subprocess
import tempfile
import numpy as np


def decode(arrays, metadata, confidence=0.25, nms=0.45):
    boxes, scores = [], []
    for branch in range(3):
        attrs = metadata['outputs'][branch * 3:branch * 3 + 3]
        box = (arrays[branch * 3].astype(np.float32) - attrs[0]['zp']) * np.float32(attrs[0]['scale'])
        score = (arrays[branch * 3 + 1].astype(np.float32) - attrs[1]['zp']) * np.float32(attrs[1]['scale'])
        height, width = score.shape[-2:]
        # Classification outputs have sigmoid applied in the graph.
        points = np.flatnonzero(score.ravel() >= np.float32(confidence))
        logits = box.reshape(4, 16, -1)[:, :, points].astype(np.float64)
        if not points.size:
            continue
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        distance = (weights * np.arange(16)[None, :, None]).sum(axis=1) / weights.sum(axis=1)
        stride = 416.0 / height
        cx, cy = (points % width + .5) * stride, (points // width + .5) * stride
        boxes.extend(np.stack((cx - distance[0] * stride, cy - distance[1] * stride,
                               cx + distance[2] * stride, cy + distance[3] * stride), axis=1))
        scores.extend(score.ravel()[points])
    boxes = np.asarray(boxes).reshape(-1, 4)
    scores = np.asarray(scores)
    order = np.argsort(-scores, kind='stable')
    kept = []
    while order.size:
        i, remaining = order[0], order[1:]
        kept.append(i)
        if not remaining.size:
            break
        intersection = np.maximum(0, np.minimum(boxes[i, 2:], boxes[remaining, 2:]) -
                                  np.maximum(boxes[i, :2], boxes[remaining, :2])).prod(axis=1)
        area_i = np.maximum(0, boxes[i, 2:] - boxes[i, :2]).prod()
        area_remaining = np.maximum(0, boxes[remaining, 2:] - boxes[remaining, :2]).prod(axis=1)
        total = area_i + area_remaining - intersection
        overlap = np.divide(intersection, total, out=np.zeros_like(total), where=total > 0)
        order = remaining[overlap <= nms]
    width, height = metadata['width'], metadata['height']
    scale = min(416 / width, 416 / height)
    pad = np.array([(416 - round(width * scale)) / 2, (416 - round(height * scale)) / 2] * 2)
    boxes = (boxes[kept] - pad) / scale
    boxes = np.clip(boxes, 0, [width, height, width, height])
    result = np.column_stack((boxes, scores[kept]))
    return result[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]


def load(directory, confidence, nms):
    metadata = json.loads((directory / 'metadata.json').read_text())
    arrays = []
    lines = [f"{metadata['width']} {metadata['height']} {confidence} {nms}"]
    for i, attr in enumerate(metadata['outputs']):
        shape = (attr['channels'], attr['height'], attr['width'])
        arrays.append(np.fromfile(directory / f'output_{i}.bin', dtype=np.int8).reshape(shape))
        lines.append(' '.join(str(attr[k]) for k in ('channels', 'height', 'width', 'zp', 'scale')))
    (directory / 'contract.txt').write_text('\n'.join(lines) + '\n')
    return arrays, metadata


def compare(directory, executable, confidence, nms, jsonl=None):
    arrays, metadata = load(directory, confidence, nms)
    expected = decode(arrays, metadata, confidence, nms)
    actual = np.asarray(json.loads(subprocess.check_output([str(executable), str(directory)], text=True))).reshape(-1, 5)
    if expected.shape != actual.shape:
        raise AssertionError(f'detection count mismatch: {expected.shape} versus {actual.shape}')
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-3)
    if jsonl:
        with open(jsonl, encoding='utf8') as file:
            first = json.loads(next(file))
        emitted = np.asarray([d['box'] + [d['score']] for d in first['detections']]).reshape(-1, 5)
        np.testing.assert_allclose(emitted, expected, rtol=1e-5, atol=2e-3)
    return len(actual)


def synthetic(directory, width, height, seed):
    rng = np.random.default_rng(seed)
    metadata = {'width': width, 'height': height, 'outputs': []}
    for i in range(9):
        side, channels = 52 >> (i // 3), 64 if i % 3 == 0 else 1
        attr = dict(channels=channels, height=side, width=side, zp=-128 if i % 3 else 3,
                    scale=1 / 255 if i % 3 else .07)
        if i % 3 == 0:
            data = rng.integers(-128, 128, size=(channels, side, side), dtype=np.int8)
        else:
            data = np.full((1, side, side), -128, dtype=np.int8)
            if i % 3 == 1:
                points = rng.choice(side * side, min(35, side * side), replace=False)
                data.ravel()[points] = rng.integers(-65, 120, size=len(points), dtype=np.int8)
        data.tofile(directory / f'output_{i}.bin')
        metadata['outputs'].append(attr)
    (directory / 'metadata.json').write_text(json.dumps(metadata))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decoder', type=pathlib.Path, required=True)
    parser.add_argument('--dump', type=pathlib.Path)
    parser.add_argument('--jsonl')
    parser.add_argument('--conf', type=float, default=.25)
    parser.add_argument('--nms', type=float, default=.45)
    args = parser.parse_args()
    if args.dump:
        print(f'Raw output reference passed: {compare(args.dump, args.decoder, args.conf, args.nms, args.jsonl)} people')
    else:
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            for seed, size in enumerate([(416, 416), (1920, 1080), (1080, 1920), (641, 479), (479, 641), (832, 421)]):
                synthetic(directory, *size, seed)
                compare(directory, args.decoder, args.conf, args.nms)
        print('NumPy/C++ reference passed: 6 sizes, three output scales, random INT8 logits')


if __name__ == '__main__':
    main()
