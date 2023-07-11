# DiamondWire


# How to Usage

### Train

```bash
python train.py -b 16 -lr 0.001 -num_workers 4
```

### Test

```bash
python test.py -b 16 -lr 0.001 -num_workers 4
```
#### 옵션 정리

- -b : batch size
- -lr : learning rate
- -num_workers : pytorch DataLoader에서 데이터를 로드하는 동안 사용할 서브 프로세스의 수

### Clustering, Calculate PSNR, SSIM, MSE
- 같은 특징을 갖는 데이터들끼리 묶고, 3차원 그래프로 시각화
- 또한 허프 변환 직선 알고리즘으로 그어진 직선 와이어와 원본 와이어와의 유사도를 비교하여 high, medium, low level의 정도를 결정한다.

### Split_reigion
- 와이어 데이터는 상단, 중단, 하단으로 나눠서 볼 수 있는데 각 영역에 따라 high, medium, low level의 정도가 다르다.
- 따라서, 허프 직선 변환 알고리즘을 통해 와이어 데이터를 잘 나타내는 직선을 그은 뒤 그에 맞게 영역을 분할하고 비교한다.
