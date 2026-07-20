import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


class WaferDataset(Dataset):
    """npz로 저장된 웨이퍼맵을 3채널 one-hot으로 바꿔서 리턴.

    0/1/2 값을 그대로 1채널로 넣어봤는데 배경(0)과 정상다이(1)의 경계가
    희미해져서인지 성능이 애매했음. 채널을 나눠주니 좀 나았다.
    """

    def __init__(self, npz_path, augment=False):
        d = np.load(npz_path)
        self.X = d['X']  # (N, H, W) uint8, 값은 0/1/2
        self.y = d['y']
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        wm = self.X[idx]

        if self.augment:
            # 웨이퍼는 원형이라 90도 회전/뒤집기 해도 패턴 클래스가 안 변함
            # (Edge-Ring 돌려도 Edge-Ring임)
            k = np.random.randint(4)
            wm = np.rot90(wm, k)
            if np.random.rand() < 0.5:
                wm = np.fliplr(wm)
            wm = wm.copy()  # 음수 stride 때문에 copy 필요

        # one-hot: (3, H, W) = [배경, 정상다이, 불량다이]
        x = np.stack([(wm == 0), (wm == 1), (wm == 2)]).astype(np.float32)
        return torch.from_numpy(x), int(self.y[idx])


def make_sampler(y, num_classes=9):
    # 클래스 불균형이 심해서 (none 13000 vs Near-full 149)
    # 배치 안에서 비율을 맞춰주는 sampler 사용
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    weights = 1.0 / np.clip(counts, 1, None)
    sample_w = weights[y]
    return WeightedRandomSampler(
        torch.from_numpy(sample_w), num_samples=len(y), replacement=True)
