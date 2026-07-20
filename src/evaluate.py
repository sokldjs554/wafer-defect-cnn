# 테스트셋 최종 평가 + confusion matrix / 오분류 샘플 저장

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from dataset import WaferDataset
from model import WaferCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/processed')
    parser.add_argument('--model', default='results/best_model.pt')
    parser.add_argument('--out', default='results')
    args = parser.parse_args()

    with open(os.path.join(args.data, 'meta.json')) as f:
        classes = json.load(f)['classes']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WaferCNN().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    test_ds = WaferDataset(os.path.join(args.data, 'test.npz'))
    loader = DataLoader(test_ds, batch_size=256)

    preds = []
    with torch.no_grad():
        for x, _ in loader:
            preds.append(model(x.to(device)).argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    labels = test_ds.y

    report = classification_report(labels, preds, target_names=classes,
                                   digits=4)
    print(report)
    with open(os.path.join(args.out, 'test_report.txt'), 'w') as f:
        f.write(report)

    # confusion matrix (행 기준 정규화 - 클래스별 샘플수 차이가 커서)
    cm = confusion_matrix(labels, preds, normalize='true')
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center',
                        fontsize=7,
                        color='white' if cm[i, j] > 0.5 else 'black')
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'confusion_matrix.png'), dpi=150)

    # 오분류 샘플 몇 개 뽑아서 확인용으로 저장
    wrong = np.where(preds != labels)[0]
    if len(wrong) > 0:
        pick = wrong[:16]
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for ax_, idx in zip(axes.flat, pick):
            ax_.imshow(test_ds.X[idx], cmap='viridis')
            ax_.set_title(f'true {classes[labels[idx]]}\n'
                          f'pred {classes[preds[idx]]}', fontsize=8)
            ax_.axis('off')
        for ax_ in axes.flat[len(pick):]:
            ax_.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, 'misclassified.png'), dpi=150)
        print(f'오분류 {len(wrong)}/{len(labels)}')


if __name__ == '__main__':
    main()
