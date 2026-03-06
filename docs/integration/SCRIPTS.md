# `scripts/`

scan2sim utilities are exposed through one CLI.

- Entry CLI: `python -m scripts <command>`
- Runtime I/O root: `outputs/`
- LiveHPS upstream source is referenced from: `../LiveHPS` (not copied into scan2sim)

Canonical pipeline commands:
- `pcap2pcd`: `raw.pcap -> data/real/raw/*.pcd`
- `pcd2human`: `data/real/raw/*.pcd -> data/real/human/human_###.ply`
- `human2smpl`: `human_###.ply -> outputs/smpl/livehps_smpl_###.npz`
- `npz2quat`, `npz2obj`, `quat2unity`, `unity2label`
- `build-feature-manifest`: point cloud + label json -> `data/feature/manifest.jsonl`
- `train-feature`: PointNet++ feature-only multi-head train
- `infer-feature`: PointNet++ feature-only multi-head infer
- `pcd2ply`: standalone format conversion (`.pcd -> .ply`)
- `run-livehps`: LiveHPS end-to-end 실행
- `run-feature-pipeline`: PointNet++ manifest/train/infer 일괄 실행
- `run-feature`: PointNet++ feature-only 추론 실행

PointNet++ train note:
- `train-feature`는 기본으로 `weights/pointnet2/.../pointnet2_ssg_wo_normals_best_model.pth`를 encoder 초기화에 사용하고, 12-head는 새로 학습합니다.

Pipeline modules:
- `scripts/pipeline/run_livehps_pipeline.py`
- `scripts/pipeline/run_feature_pipeline.py`
- `scripts/infer/infer_feature_classifier.py` (feature backend)

Equivalent long-form names are also supported:
- `pcap-to-pcd`, `pcd-to-human`, `human-to-smpl`
- `pcd-to-ply`, `npz-to-quat`, `npz-to-obj`, `quat-to-unity`, `unity-to-label`
