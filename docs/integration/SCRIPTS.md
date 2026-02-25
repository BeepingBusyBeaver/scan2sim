# `scripts/`

LiveHPS-related utilities are merged into scan2sim script groups and exposed through one CLI.

- Entry CLI: `python -m scripts <command>`
- Runtime I/O root: `outputs/`
- Upstream source is referenced from: `../LiveHPS` (not copied into scan2sim)

Canonical pipeline commands:
- `pcap2pcd`: `raw.pcap -> data/real/raw/*.pcd`
- `pcd2human`: `data/real/raw/*.pcd -> data/real/human/human_###.ply`
- `human2smpl`: `human_###.ply -> outputs/smpl/livehps_smpl_###.npz`
- `npz2quat`, `npz2obj`, `quat2unity`, `unity2label`
- `run-pipeline`: run end-to-end pipeline in one command

Pipeline runner module:
- `scripts/pipeline/run_livehps_pipeline.py`

Equivalent long-form names are also supported:
- `pcap-to-pcd`, `pcd-to-human`, `human-to-smpl`
- `npz-to-quat`, `npz-to-obj`, `quat-to-unity`, `unity-to-label`
