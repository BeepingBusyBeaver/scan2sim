# scan2sim Project Structure

## Canonical folders
- `configs/`: project configuration files (`axis_steps`, `feature/pointnet2_multitask.yaml`, `feature/label_rules.json`, `parser/*.yaml`)
- `data/`: canonical runtime data root
  - `data/background/`: background reference + selected frames
  - `data/interim/`: intermediate processing outputs (e.g., person extraction)
  - `data/labeling/`: pcap/csv/fbx and labeling resources
  - `data/real/raw/`, `data/real/human/`, `data/real/VAR/`, `data/real/VARmotion/`
  - `data/feature/`: feature-only 학습용 manifest
  - `data/match_one/`: match-one evaluation result artifacts
  - `data/virtual/`: virtual pose libraries
- `outputs/`: interop runtime I/O (`smpl`, `quat`, `euler`, `label`, `obj`, `feature_models`, `parser_label`)
- `dataset/`: pose/label definition datasets (json/jsonl)
- `scripts/`: scan2sim processing utilities (`common`, `data`, `ouster`, `pc`, `conversion`, `labeling`, `infer`, `feature`, `parser`, `train`, `eval`, `pipeline`, `workspace`)
- `tests/`: test assets and validation scripts

## Path defaults source
- Canonical CLI input/output defaults are defined in `scripts/common/io_paths.py`.
- Scripts should reference this module instead of hardcoding path strings.

## Coordinate policy
- Shared coordinate/rotation conversion helpers live in `scripts/common/coord_utils.py`.
- Pipeline policy is fixed as:
  - `pcap/pcd -> ply`: keep source coordinates
  - `ply -> npz`: keep source coordinates
  - `npz -> quat`: keep source coordinates
  - `quat -> unity`: export Unity Inspector Rotation XYZ (SMPL local joint 기준, 기본 `basis-conversion=none`)

## Output policy
- `data/match_one/`: preferred directory for match-one evaluation artifacts
- `outputs/feature_models/`: PointNet++ checkpoint outputs
- `outputs/parser_label/`: parser-decoder label outputs
- `output/`: legacy compatibility path (existing pipelines may still reference it)

## Backend policy
- Upstream LiveHPS code/assets stay in `/home/victus/projects/LiveHPS`.
- LiveHPS 실행은 `run-livehps`로 수행.
- Feature-only PointNet++ 실행은 `run-feature-pipeline`(일괄) 또는 `run-feature`/`train-feature`/`infer-feature`.
- Rule-based parser-decoder 실행은 `run-parser-pipeline`(일괄) 또는 `run-parser-decoder`.
- scan2sim runtime artifacts are managed under `outputs/`.
