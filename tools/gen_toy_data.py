# 합성 웨이퍼맵 생성기
# WM-811K(2GB) 안 받고도 파이프라인이 도는지 확인하려고 만든 스크립트.
# 실제 실험은 무조건 진짜 데이터로 할 것. 이건 어디까지나 smoke test용.

import argparse
import os

import numpy as np

CLASSES = ['none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
           'Loc', 'Random', 'Scratch', 'Near-full']


def base_wafer(size, rng):
    # 원형 웨이퍼: 바깥 0, 안쪽 정상다이 1
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    wm = (r <= c).astype(np.uint8)
    return wm, r, yy, xx, c


def make_map(cls, size, rng):
    wm, r, yy, xx, c = base_wafer(size, rng)
    inside = wm == 1
    defect = np.zeros_like(wm, dtype=bool)

    if cls == 'none':
        pass
    elif cls == 'Center':
        defect = r < c * rng.uniform(0.25, 0.45)
    elif cls == 'Donut':
        r1 = c * rng.uniform(0.35, 0.5)
        defect = (r > r1) & (r < r1 + c * rng.uniform(0.15, 0.25))
    elif cls == 'Edge-Loc':
        theta = np.arctan2(yy - c, xx - c)
        t0 = rng.uniform(-np.pi, np.pi)
        span = rng.uniform(0.4, 1.2)
        d = np.angle(np.exp(1j * (theta - t0)))
        defect = (r > c * 0.8) & (np.abs(d) < span / 2)
    elif cls == 'Edge-Ring':
        defect = r > c * rng.uniform(0.82, 0.9)
    elif cls == 'Loc':
        cy = c + rng.uniform(-0.4, 0.4) * c
        cx = c + rng.uniform(-0.4, 0.4) * c
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        defect = rr < c * rng.uniform(0.12, 0.25)
    elif cls == 'Random':
        defect = rng.random(wm.shape) < rng.uniform(0.08, 0.18)
    elif cls == 'Scratch':
        # 대충 직선 스크래치
        t = np.linspace(0, 1, size * 3)
        x0, y0 = rng.uniform(0.1, 0.9, 2) * size
        ang = rng.uniform(0, np.pi)
        L = rng.uniform(0.4, 0.9) * size
        xs = (x0 + np.cos(ang) * (t - 0.5) * L).astype(int)
        ys = (y0 + np.sin(ang) * (t - 0.5) * L).astype(int)
        ok = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
        defect[ys[ok], xs[ok]] = True
    elif cls == 'Near-full':
        defect = rng.random(wm.shape) < rng.uniform(0.6, 0.85)

    # 공통 노이즈
    defect |= rng.random(wm.shape) < 0.015
    wm[inside & defect] = 2
    return wm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/processed')
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--per-class', type=int, default=60)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X, y = [], []
    for ci, cls in enumerate(CLASSES):
        for _ in range(args.per_class):
            X.append(make_map(cls, args.size, rng))
            y.append(ci)
    X = np.stack(X)
    y = np.array(y, dtype=np.int64)

    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    n1, n2 = int(len(y) * 0.7), int(len(y) * 0.85)

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, 'train.npz'),
                        X=X[:n1], y=y[:n1])
    np.savez_compressed(os.path.join(args.out, 'val.npz'),
                        X=X[n1:n2], y=y[n1:n2])
    np.savez_compressed(os.path.join(args.out, 'test.npz'),
                        X=X[n2:], y=y[n2:])

    import json
    meta = {'classes': CLASSES, 'size': args.size,
            'counts': {'train': np.bincount(y[:n1], minlength=9).tolist(),
                       'val': np.bincount(y[n1:n2], minlength=9).tolist(),
                       'test': np.bincount(y[n2:], minlength=9).tolist()}}
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'합성 데이터 {len(y)}장 저장 -> {args.out}')


if __name__ == '__main__':
    main()
