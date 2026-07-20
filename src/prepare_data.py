# WM-811K 원본 pkl -> 학습용 npz 변환 스크립트
# 원본이 2GB 정도라 로딩에 시간이 좀 걸림 (램 8GB 이하면 좀 위험)

import argparse
import json
import os

import numpy as np
import pandas as pd
from skimage.transform import resize
from sklearn.model_selection import train_test_split

CLASSES = ['none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
           'Loc', 'Random', 'Scratch', 'Near-full']


def get_label(x):
    # failureType이 [['Center']] 같은 중첩 배열 형태라서 꺼내줘야 함
    # 라벨 없는 행은 빈 배열
    if isinstance(x, str):
        return x
    try:
        if len(x) == 0:
            return None
        return str(x[0][0])
    except (TypeError, IndexError):
        return None


def resize_map(wm, size):
    # 웨이퍼맵 값이 0(배경)/1(정상)/2(불량) 3가지뿐이라
    # 보간 들어가면 1.3 같은 애매한 값이 생김 -> nearest(order=0)로 리사이즈
    out = resize(wm, (size, size), order=0, preserve_range=True,
                 anti_aliasing=False)
    return out.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', default='data/LSWMD.pkl')
    parser.add_argument('--out', default='data/processed')
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--none-cap', type=int, default=13000,
                        help='none 클래스가 14.7만장이라 이만큼만 샘플링')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print('pkl 로딩중... (2GB라 1~2분 걸림)')
    df = pd.read_pickle(args.pkl)
    print(f'전체 {len(df)}장')

    df['label'] = df['failureType'].apply(get_label)
    df = df[df['label'].isin(CLASSES)].reset_index(drop=True)
    print(f'라벨 있는 것 {len(df)}장')
    print(df['label'].value_counts())

    # none이 압도적으로 많아서 cap 걸고 나머지는 버림
    rng = np.random.default_rng(args.seed)
    none_idx = df.index[df['label'] == 'none'].to_numpy()
    if len(none_idx) > args.none_cap:
        keep_none = rng.choice(none_idx, args.none_cap, replace=False)
        drop = set(none_idx) - set(keep_none)
        df = df.drop(index=list(drop)).reset_index(drop=True)
    print(f'none 샘플링 후 {len(df)}장')

    X = np.zeros((len(df), args.size, args.size), dtype=np.uint8)
    y = np.zeros(len(df), dtype=np.int64)
    for i, row in enumerate(df.itertuples()):
        X[i] = resize_map(row.waferMap, args.size)
        y[i] = CLASSES.index(row.label)
        if i % 5000 == 0:
            print(f'  resize {i}/{len(df)}')

    # 클래스 비율 유지하면서 70/15/15 분할
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=args.seed)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=args.seed)

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, 'train.npz'), X=X_tr, y=y_tr)
    np.savez_compressed(os.path.join(args.out, 'val.npz'), X=X_val, y=y_val)
    np.savez_compressed(os.path.join(args.out, 'test.npz'), X=X_te, y=y_te)

    meta = {
        'classes': CLASSES,
        'size': args.size,
        'counts': {
            'train': np.bincount(y_tr, minlength=9).tolist(),
            'val': np.bincount(y_val, minlength=9).tolist(),
            'test': np.bincount(y_te, minlength=9).tolist(),
        },
    }
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print('저장 완료:', args.out)
    print('train/val/test =', len(y_tr), len(y_val), len(y_te))


if __name__ == '__main__':
    main()
