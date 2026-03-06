# PointNet++ Weights Layout

`scan2sim`에서 PointNet++ 관련 가중치는 아래 구조로 관리합니다.

```
weights/pointnet2/
  yanx27/
    modelnet40_cls/
      pointnet2_msg_normals_best_model.pth
      pointnet2_ssg_wo_normals_best_model.pth
```

- `yanx27/`: 출처(원본 저장소) 기준 분류
- `modelnet40_cls/`: 데이터셋/태스크 기준 분류

주의:
- 위 파일은 외부 사전학습 모델(원본 PointNet++ 분류)입니다.
- `scripts/infer/infer_feature_classifier.py`의 `--checkpoint`는 `scan2sim` 학습 포맷(`checkpoint_best.pt`)을 기대합니다.
