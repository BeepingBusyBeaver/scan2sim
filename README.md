# scan2sim

LiDAR point cloud 기반 사람 추출/추론/라벨링을 위한 실행 프로젝트입니다.  
현재 `scan2sim`은 세 가지 경로를 지원합니다.
- LiveHPS 기반 SMPL 파이프라인 (`../LiveHPS` 필요)
- PointNet++ 기반 feature-only 15-head 분류 파이프라인
- Rule-based parser-decoder 파이프라인 (10개 body-part + relation features)

## 1) Requirements
- OS: Linux (WSL 포함)
- Python: `3.11`
- GPU 권장 (CUDA): `requirements-cu130.txt` 기준
- 별도 LiveHPS 리포지토리 필요:
  - 경로: `../LiveHPS`
  - 가중치: `../LiveHPS/save_models/livehps.t7`

## 2) Setup
```bash
conda create -n scan2sim python=3.11 -y
conda activate scan2sim
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-cu130.txt
```

`environment.yml`을 쓰는 경우:
```bash
conda env create -f environment.yml
conda activate scan2sim
```

## 3) CLI Usage
명령 목록:
```bash
python -m scripts
```

개별 모듈 help:
```bash
python -m scripts run-livehps --help
python -m scripts run-feature-pipeline --help
python -m scripts run-feature --help
python -m scripts run-parser-pipeline --help
python -m scripts run-parser-decoder --help
python -m scripts.pc.extract_person --help
python -m scripts.eval.match_one --help
```

## 4) LiveHPS End-to-End Pipeline
```bash
python -m scripts run-livehps \
  --pcap data/labeling/001/simpleMotion_wonjin.pcap \
  --base SN001 \
  --scan 861 893 929 \
  --raw-dir data/real/raw \
  --human-dir data/real/human \
  --smpl-dir outputs/smpl \
  --quat-dir outputs/quat \
  --euler-dir outputs/euler \
  --label-dir outputs/label \
  --obj-dir outputs/obj
```

파이프라인 단계:
1. `pcap2pcd`: `raw.pcap -> raw.pcd`
2. `pcd2human`: `raw.pcd -> human.ply`
3. `human2smpl`: `human.ply -> livehps_smpl_*.npz`
4. `npz2quat`, `npz2obj`, `npz2fbx`, `quat2unity`, `unity2label`

## 5) LiveHPS Step-by-Step Commands
```bash
python -m scripts pcap2pcd data/labeling/001/simpleMotion_wonjin.pcap \
  --scan 861 --outdir data/real/raw --base SN001

python -m scripts pcd2human \
  --in "data/real/raw/SN001_*.pcd" \
  --bg data/real/raw/SN001_bg.pcd \
  --out_dir data/real/human

python -m scripts human2smpl \
  --input data/real/human \
  --batch_per_file \
  --input_pattern "human_*.ply" \
  --output_dir outputs/smpl \
  --output_prefix "livehps_smpl_"

python -m scripts npz2quat --input outputs/smpl --batch_per_file
python -m scripts npz2obj --input outputs/smpl --output outputs/obj
python -m scripts npz2fbx \
  --input "outputs/smpl/livehps_smpl_*.npz" \
  --output outputs/fbx/livehps_rigged_full.fbx \
  --blender "/mnt/c/Program Files/Blender Foundation/Blender 4.3/blender.exe"
python -m scripts quat2unity --input outputs/quat --batch_per_file
python -m scripts unity2label --batch_per_file --euler-json outputs/euler --quat-json outputs/quat
```

## 6) Feature-Only 15-Head (PointNet++)
`Euler/quat/SMPL` 계산 없이 point cloud feature만으로 15개 head를 분류합니다.

원클릭 파이프라인(매니페스트 생성 → 학습 → 추론):
```bash
python -m scripts run-feature-pipeline \
  --points-dir data/real/VAR001/human \
  --labels-dir data/feature/labels
```

1) point cloud + label JSON에서 manifest 생성
```bash
python -m scripts build-feature-manifest \
  --points-dir data/real/VAR001/human \
  --point-pattern "human_*.ply" \
  --labels-dir data/feature/labels \
  --label-pattern "*.json" \
  --output data/feature/manifest.jsonl
```

2) PointNet++ multi-head 학습
```bash
python -m scripts train-feature \
  --config configs/feature/pointnet2_multitask.yaml
```

기본값으로 `weights/pointnet2/yanx27/modelnet40_cls/pointnet2_ssg_wo_normals_best_model.pth`를
encoder 초기화에 자동 사용하고, head는 scan2sim에서 새로 학습합니다.  
기본 학습 모드는 backbone freeze + head 교체(`freeze_backbone=true`)입니다.
자동 초기화를 끄려면 `--no-auto-init-backbone`을 사용하세요.

3) 학습 모델 추론
```bash
python -m scripts infer-feature \
  --checkpoint outputs/feature_models/pointnet2_15head/checkpoint_best.pt \
  --input data/real/VAR001/human \
  --input-pattern "human_*.ply" \
  --output-dir outputs/feature_pred
```

`--checkpoint`를 생략하면 `outputs/feature_models/...`와 `weights/pointnet2/...`를 자동 탐색합니다.

동일 기능은 직접 추론 명령으로도 호출:
```bash
python -m scripts run-feature \
  --checkpoint outputs/feature_models/pointnet2_15head/checkpoint_best.pt \
  --input data/real/VAR001/human \
  --input-pattern "human_*.ply"
```

## 7) Rule-Based Parser-Decoder
PointNet++ 분류기와 별도 트랙으로, 점군을 10개 body-part 박스로 파싱한 후
part relation만으로 15개 라벨을 복원합니다.

```bash
python -m scripts run-parser-pipeline \
  --input data/real/SQUAT/human \
  --input-pattern "human_*.ply" \
  --profile squat \
  --output-dir outputs/SQUAT/parser_label \
  --gt-jsonl data/feature/SQUAT_label_GT.jsonl
```

직접 실행(단일 명령):
```bash
python -m scripts run-parser-decoder \
  --input data/real/SQUAT/human \
  --input-pattern "human_*.ply" \
  --profile squat \
  --part-config configs/parser/part_parser_rules.yaml \
  --decoder-config configs/parser/label_decoder_rules.yaml \
  --head-rules configs/feature/label_rules.json \
  --output-dir outputs/SQUAT/parser_label
```

`--profile` 지원: `squat`, `lunge`, `wbmotion`  
(profile별로 temporal/consistency/swap-validation 규칙이 분리됨)

## 8) Match One (Real ↔ Virtual)
```bash
python -m scripts.eval.match_one \
  --real "data/real/human/human_001.ply" \
  --virtual-glob "data/virtual/pose324_10_deg45/virtual_*.ply" \
  --poses-jsonl dataset/324_LowerBody_10set.jsonl \
  --out-root data/match_one \
  --topk 3
```

## 9) Path Policy
- 기본 경로 상수: `scripts/common/io_paths.py`
- 모든 상대 경로는 `scan2sim/` 루트 기준으로 해석
- 런타임 산출물:
  - `outputs/`: SMPL/quat/euler/label/obj/feature_models
  - `outputs/parser_label/`: parser-decoder 산출물
  - `data/feature/manifest.jsonl`: feature 학습용 샘플 매니페스트
  - `data/match_one/`: 매칭 평가 산출물

## 10) Related Docs
- `docs/PROJECT_STRUCTURE.md`
- `docs/integration/SCRIPTS.md`
