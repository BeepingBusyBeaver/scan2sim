# scan2sim

LiDAR point cloud 기반 사람 추출/SMPL 추론/라벨링을 위한 실행 프로젝트입니다.  
추론 모델은 `../LiveHPS`(공식 클론 + 가중치)를 사용하고, `scan2sim`은 파이프라인/변환/평가를 담당합니다.

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
python -m scripts.pipeline.run_livehps_pipeline --help
python -m scripts.pc.extract_person --help
python -m scripts.eval.match_one --help
```

## 4) End-to-End Pipeline (권장)
```bash
python -m scripts run-pipeline \
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
4. `npz2quat`, `npz2obj`, `quat2unity`, `unity2label`

## 5) Step-by-Step Commands
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
python -m scripts quat2unity --input outputs/quat --batch_per_file
python -m scripts unity2label --batch_per_file --euler-json outputs/euler --quat-json outputs/quat
```

## 6) Match One (Real ↔ Virtual)
```bash
python -m scripts.eval.match_one \
  --real "data/real/human/human_001.ply" \
  --virtual-glob "data/virtual/pose324_10_deg45/virtual_*.ply" \
  --poses-jsonl dataset/324_LowerBody_10set.jsonl \
  --out-root data/match_one \
  --topk 3
```

## 7) Path Policy
- 기본 경로 상수: `scripts/common/io_paths.py`
- 모든 상대 경로는 `scan2sim/` 루트 기준으로 해석
- 런타임 산출물:
  - `outputs/`: SMPL/quat/euler/label/obj
  - `data/match_one/`: 매칭 평가 산출물

## 8) Related Docs
- `docs/PROJECT_STRUCTURE.md`
- `docs/integration/SCRIPTS.md`
- `integrations/livehps/README.md`
