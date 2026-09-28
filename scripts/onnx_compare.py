#!/usr/bin/env python3
"""Host FP32 ONNX / board JSONL comparison. This checks consistency, not ground-truth accuracy."""
import argparse
import hashlib
import json
import pathlib
import statistics
import cv2
import numpy as np
import onnxruntime as ort
from compare_jsonl import overlap
from reference_decode import decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', type=pathlib.Path, required=True)
    parser.add_argument('--video', type=pathlib.Path, required=True)
    parser.add_argument('--board-jsonl', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--conf', type=float, default=.25)
    parser.add_argument('--nms', type=float, default=.45)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1 or not (0 < args.conf <= 1 and 0 <= args.nms <= 1):
        parser.error('Invalid threads or thresholds')
    for source in (args.onnx, args.video, args.board_jsonl):
        if not source.is_file():
            parser.error(f'File does not exist: {source}')
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=['CPUExecutionProvider'])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    expected_shapes = [[1, 64 if i % 3 == 0 else 1, 52 >> (i // 3), 52 >> (i // 3)] for i in range(9)]
    if len(inputs) != 1 or inputs[0].shape != [1, 3, 416, 416] or inputs[0].type != 'tensor(float)':
        raise ValueError('Expected FP32 ONNX with input [1,3,416,416]')
    if len(outputs) != 9 or [o.shape for o in outputs] != expected_shapes:
        raise ValueError('Expected nine DFL/person/score_sum outputs at 52/26/13')
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise ValueError('Cannot open video')
    stats = dict(frames=0, board_detections=0, onnx_detections=0, board_empty_frames=0,
                 onnx_empty_frames=0, same_count_frames=0, board_large_boxes_over_20pct=0,
                 onnx_large_boxes_over_20pct=0, matched_boxes_iou_at_least_0_5=0)
    examples, matched_ious, score_deltas = [], [], []
    try:
        with args.board_jsonl.open(encoding='utf8') as board, (args.out / 'onnx.jsonl').open('w', encoding='utf8') as reference:
            for frame_id, line in enumerate(board):
                row = json.loads(line)
                if row['frame_id'] != frame_id:
                    raise ValueError(f'Board JSONL has a frame gap/repetition at {frame_id}')
                ok, image = cap.read()
                if not ok:
                    raise ValueError('Video is shorter than the board JSONL')
                height, width = image.shape[:2]
                scale = min(416 / width, 416 / height)
                nw, nh = round(width * scale), round(height * scale)
                left, top = (416 - nw) // 2, (416 - nh) // 2
                rgb = np.zeros((416, 416, 3), dtype=np.uint8)
                rgb[top:top + nh, left:left + nw] = cv2.cvtColor(cv2.resize(image, (nw, nh)), cv2.COLOR_BGR2RGB)
                array = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255
                tensors = session.run(None, {inputs[0].name: array})
                attributes = [dict(channels=a.shape[1], height=a.shape[2], width=a.shape[3], zp=0, scale=1) for a in tensors]
                predicted = decode(tensors, dict(width=width, height=height, outputs=attributes), args.conf, args.nms)
                detections = [dict(class_id=0, label='person', box=d[:4].tolist(), score=float(d[4])) for d in predicted]
                result = dict(frame_id=frame_id, source=row['source'], timestamp_ms=row['timestamp_ms'], detections=detections)
                reference.write(json.dumps(result) + '\n')
                stats['frames'] += 1
                stats['same_count_frames'] += len(row['detections']) == len(detections)
                large = {}
                for name, ds in [('board', row['detections']), ('onnx', detections)]:
                    stats[name + '_detections'] += len(ds)
                    stats[name + '_empty_frames'] += not ds
                    large[name] = sum((d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1]) > .2 * width * height for d in ds)
                    stats[name + '_large_boxes_over_20pct'] += large[name]
                if len(examples) < 8 and large['board'] and large['onnx']:
                    examples.append(dict(frame_id=frame_id, board=row['detections'], onnx=detections))
                remaining = list(detections)
                for d in row['detections']:
                    if not remaining:
                        break
                    value, index = max((overlap(d['box'], r['box']), i) for i, r in enumerate(remaining))
                    if value >= .5:
                        matched = remaining.pop(index)
                        matched_ious.append(value)
                        score_deltas.append(abs(d['score'] - matched['score']))
            if cap.read()[0]:
                raise ValueError('Video is longer than the board JSONL; compare a complete run')
    finally:
        cap.release()
    if not stats['frames']:
        raise ValueError('Empty comparison input')
    stats['matched_boxes_iou_at_least_0_5'] = len(matched_ious)
    stats['matched_median_iou'] = statistics.median(matched_ious) if matched_ious else None
    stats['matched_median_score_difference'] = statistics.median(score_deltas) if score_deltas else None
    stats['examples_of_large_boxes_in_both'] = examples
    stats['onnx_sha256'] = hashlib.sha256(args.onnx.read_bytes()).hexdigest()
    stats['video_sha256'] = hashlib.sha256(args.video.read_bytes()).hexdigest()
    stats['confidence'] = args.conf
    stats['nms'] = args.nms
    stats['ground_truth_used'] = False
    stats['note'] = 'ONNX is a consistency reference, not ground truth. These values are not precision, recall or mAP.'
    (args.out / 'comparison.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps({k: v for k, v in stats.items() if k != 'examples_of_large_boxes_in_both'}, indent=2))


if __name__ == '__main__':
    main()
