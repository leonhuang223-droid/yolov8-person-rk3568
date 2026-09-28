#!/usr/bin/env python3
import argparse
import json

parser = argparse.ArgumentParser(description='Check consecutive frame IDs, person-only boxes, and optional frame count')
parser.add_argument('jsonl')
parser.add_argument('--frames', type=int)
args = parser.parse_args()
count = 0
with open(args.jsonl, encoding='utf8') as file:
    for count, line in enumerate(file, 1):
        row = json.loads(line)
        if row['frame_id'] != count - 1:
            raise ValueError(f'Frame gap/repetition at line {count}')
        for d in row['detections']:
            x1, y1, x2, y2 = d['box']
            if d['class_id'] != 0 or d['label'] != 'person' or not (0 <= x1 < x2 and 0 <= y1 < y2):
                raise ValueError(f'Invalid person detection at frame {count - 1}')
if args.frames is not None and count != args.frames:
    raise ValueError(f'Expected {args.frames} frames, found {count}')
print(f'JSONL passed: {count} consecutive frames')
