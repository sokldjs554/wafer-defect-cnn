# data

용량 문제로 데이터는 git에 안 올림.

1. https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map 에서 다운로드
2. `LSWMD.pkl`을 이 폴더에 두기
3. 루트에서 `python src/prepare_data.py --pkl data/LSWMD.pkl` 실행하면
   `data/processed/`에 train/val/test npz가 생김
