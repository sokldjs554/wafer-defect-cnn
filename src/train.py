import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from dataset import WaferDataset, make_sampler
from model import WaferCNN


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        out = model(x.to(device))
        preds.append(out.argmax(1).cpu().numpy())
        labels.append(y.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    acc = (preds == labels).mean()
    macro_f1 = f1_score(labels, preds, average='macro')
    return acc, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/processed')
    parser.add_argument('--out', default='results')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke', action='store_true',
                        help='동작 확인용. 1 epoch만 돌림')
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    train_ds = WaferDataset(os.path.join(args.data, 'train.npz'), augment=True)
    val_ds = WaferDataset(os.path.join(args.data, 'val.npz'))
    print(f'train {len(train_ds)} / val {len(val_ds)}')

    sampler = make_sampler(train_ds.y)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=256, num_workers=2)

    model = WaferCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    epochs = 1 if args.smoke else args.epochs
    os.makedirs(args.out, exist_ok=True)
    history = {'train_loss': [], 'val_acc': [], 'val_f1': []}
    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        val_acc, val_f1 = evaluate(model, val_loader, device)
        history['train_loss'].append(float(np.mean(losses)))
        history['val_acc'].append(float(val_acc))
        history['val_f1'].append(float(val_f1))

        # 정확도 대신 macro F1 기준으로 best 저장
        # (불균형 데이터라 acc는 none만 잘 맞춰도 높게 나옴)
        mark = ''
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(),
                       os.path.join(args.out, 'best_model.pt'))
            mark = ' *'

        print(f'[{epoch+1:2d}/{epochs}] loss {np.mean(losses):.4f} | '
              f'val acc {val_acc:.4f} | val macro-F1 {val_f1:.4f} | '
              f'{time.time()-t0:.0f}s{mark}')

    with open(os.path.join(args.out, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # 학습 곡선 저장
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history['train_loss'])
    ax[0].set_title('train loss')
    ax[0].set_xlabel('epoch')
    ax[1].plot(history['val_acc'], label='val acc')
    ax[1].plot(history['val_f1'], label='val macro-F1')
    ax[1].set_xlabel('epoch')
    ax[1].legend()
    ax[1].set_title('validation')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'training_curve.png'), dpi=150)

    print(f'best val macro-F1: {best_f1:.4f}')


if __name__ == '__main__':
    main()
