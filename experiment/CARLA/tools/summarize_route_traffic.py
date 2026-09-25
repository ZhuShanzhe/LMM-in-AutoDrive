"""Summarize geometric traffic coverage without equating it to pixel visibility."""
import argparse
import json
from pathlib import Path


def summarize(rows, fps=20):
    def stats(items, key):
        values = [item[key] for item in items]
        longest = current = 0
        for value in values:
            current = current + 1 if value == 0 else 0
            longest = max(longest, current)
        return dict(min=min(values), max=max(values), mean=sum(values) / len(values),
                    zero_frames=values.count(0), longest_empty_s=longest / fps,
                    at_least_three_fraction=sum(v >= 3 for v in values) / len(values))

    if not rows or fps <= 0:
        raise ValueError('nonempty rows and positive FPS required')
    fields = ['front_count']
    if all('front_same_direction_count' in r for r in rows):
        fields.append('front_same_direction_count')
    result = dict(frames=len(rows), duration_s=len(rows) / fps,
                  route_m=rows[-1]['progress_m'],
                  counts={key: stats(rows, key) for key in fields}, segments=[])
    for start in range(0, int(rows[-1]['progress_m']) + 1, 1000):
        segment = [r for r in rows if start <= r['progress_m'] < start + 1000]
        if segment:
            result['segments'].append(dict(start_m=start, end_m=segment[-1]['progress_m'],
                counts={key: stats(segment, key) for key in fields}))
    result['scope'] = 'Camera-only geometric forward-cone counts; no occlusion test or ego driving assessment.'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=Path)
    parser.add_argument('--fps', type=float, default=20)
    args = parser.parse_args()
    with args.log.open(encoding='utf-8') as stream:
        result = summarize([json.loads(line) for line in stream], args.fps)
    output = args.log.with_name('coverage.json')
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
