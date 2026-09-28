#!/usr/bin/env python3
"""Compare complete ordered detections between standard/bound, OpenCV/RGA and worker counts."""
import argparse
import itertools
import json


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    total = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / total if total > 0 else 0


def compare(baseline, candidate, min_iou=.98, score_tolerance=.02):
    frames = 0
    with open(baseline, encoding='utf8') as left, open(candidate, encoding='utf8') as right:
        for index, pair in enumerate(itertools.zip_longest(left, right)):
            if None in pair:
                raise ValueError(f'Frame count mismatch at {index}')
            a, b = map(json.loads, pair)
            if a['frame_id'] != index or b['frame_id'] != index or a['source'] != b['source']:
                raise ValueError(f'Frame ID/source mismatch at {index}')
            remaining = list(b['detections'])
            if len(remaining) != len(a['detections']):
                raise ValueError(f'Person count mismatch at frame {index}')
            for detection in a['detections']:
                eligible = [(overlap(detection['box'], d['box']), i) for i, d in enumerate(remaining)
                            if detection['class_id'] == d['class_id'] and abs(detection['score'] - d['score']) <= score_tolerance]
                if not eligible or max(eligible)[0] < min_iou:
                    raise ValueError(f'Box/score mismatch at frame {index}')
                remaining.pop(max(eligible)[1])
            frames += 1
    return frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline')
    parser.add_argument('candidate')
    parser.add_argument('--iou', type=float, default=.98)
    parser.add_argument('--score-tolerance', type=float, default=.02)
    args = parser.parse_args()
    print(f'Detection comparison passed: {compare(args.baseline, args.candidate, args.iou, args.score_tolerance)} frames')
