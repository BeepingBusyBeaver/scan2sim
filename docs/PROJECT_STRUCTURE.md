# scan2sim Project Structure

## Canonical folders
- `configs/`: project configuration files
- `data/`: canonical runtime data root
  - `data/background/`: background reference + selected frames
  - `data/interim/`: intermediate processing outputs (e.g., person extraction)
  - `data/labeling/`: pcap/csv/fbx and labeling resources
  - `data/real/raw/`, `data/real/human/`, `data/real/VAR/`, `data/real/VARmotion/`
  - `data/match_one/`: match-one evaluation result artifacts
  - `data/virtual/`: virtual pose libraries
- `outputs/`: interop runtime I/O (`smpl`, `quat`, `euler`, `label`, `obj`)
- `dataset/`: pose/label definition datasets (json/jsonl)
- `scripts/`: scan2sim processing utilities (`common`, `data`, `ouster`, `pc`, `conversion`, `labeling`, `infer`, `eval`, `pipeline`, `workspace`)
- `tests/`: test assets and validation scripts
- `integrations/livehps/`: upstream reference overrides/snapshots only

## Path defaults source
- Canonical CLI input/output defaults are defined in `scripts/common/io_paths.py`.
- Scripts should reference this module instead of hardcoding path strings.

## Output policy
- `data/match_one/`: preferred directory for match-one evaluation artifacts
- `output/`: legacy compatibility path (existing pipelines may still reference it)

## LiveHPS integration policy
- Upstream LiveHPS code/assets stay in `/home/victus/projects/LiveHPS`.
- scan2sim interop runtime artifacts are managed under `outputs/`.
- scan2sim execution scripts are managed under `scripts`.
- `integrations/livehps` keeps override snapshots for traceability only.
