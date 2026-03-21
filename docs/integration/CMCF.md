# CMCF (Canonical Marker Correspondence Field)

## Goal
- Canonical 인체 위 marker ID를 고정하고 pose별 위치만 이동(transport)
- `Euler/quaternion/SMPL` 복원 없이 marker relation feature만으로 12-head 라벨 디코딩
- sample exact accuracy 중심 평가

## Pipeline
1. `canonical marker field`
   - canonical frame 1개 선택
   - part별 marker 샘플링(FPS)
   - `marker_id`, `part_id`, `attachment` 저장
2. `correspondence-preserving transport`
   - 프레임별 part cloud에 marker를 nearest + unique 후보로 매핑
3. `feature-only decode`
   - marker-group box -> relation feature(distance/overlap/contact/delta)
   - deterministic decoder(`configs/cmcf/decoder_rules_12head.yaml`)
4. `prototype mapping`
   - virtual prototype bank와 real query feature-map 거리 비교
   - top-k 후보 + predicted label 출력

## Commands
### Build canonical/prototypes (virtual)
```bash
python -m scripts run-cmcf-pipeline \
  --input "data/virtual/pdbr_25421_lidar/*.ply" \
  --input-pattern "*.ply" \
  --part-config configs/parser/part_parser_rules_virtual.yaml \
  --marker-config configs/cmcf/marker_field_virtual.yaml \
  --decoder-config configs/cmcf/decoder_rules_12head.yaml \
  --viz-dir outputs/cmcf/virtual_pdbr/viz \
  --viz-write-lines \
  --output-dir outputs/cmcf/virtual_pdbr
```

### Build query features (real)
```bash
python -m scripts run-cmcf-pipeline \
  --input data/real/SQUAT/human \
  --input-pattern "human_*.ply" \
  --part-config configs/parser/part_parser_rules_v1.yaml \
  --marker-config configs/cmcf/marker_field_virtual.yaml \
  --decoder-config configs/cmcf/decoder_rules_12head.yaml \
  --viz-dir outputs/cmcf/real_squat/viz \
  --viz-write-lines \
  --output-dir outputs/cmcf/real_squat
```

### Map query to virtual prototypes
```bash
python -m scripts cmcf-map \
  --prototype-bank outputs/cmcf/virtual_pdbr/prototype_bank.json \
  --query outputs/cmcf/real_squat \
  --query-pattern "cmcf_*.json" \
  --output outputs/cmcf/real_squat/prototype_mapping.jsonl \
  --topk 3
```

### Build playback + GUI launch in one command
```bash
WAYLAND_DISPLAY= DISPLAY=:0 XDG_SESSION_TYPE=x11 \
python -m scripts run-cmcf-pipeline \
  --input "data/virtual/pdbr_25421_lidar/*.ply" \
  --input-pattern "*.ply" \
  --part-config configs/parser/part_parser_rules_virtual.yaml \
  --marker-config configs/cmcf/marker_field_virtual.yaml \
  --decoder-config configs/cmcf/decoder_rules_12head.yaml \
  --viz-dir outputs/cmcf/virtual_pdbr/viz \
  --viz-write-lines \
  --build-playback \
  --play-gui \
  --play-fps 2 \
  --play-loop \
  --play-log-every 30 \
  --output-dir outputs/cmcf/virtual_pdbr
```

## Output files
- `outputs/cmcf/<exp>/canonical_markers.json`
- `outputs/cmcf/<exp>/cmcf_*.json`
- `outputs/cmcf/<exp>/cmcf_markers.jsonl`
- `outputs/cmcf/<exp>/cmcf_features.jsonl`
- `outputs/cmcf/<exp>/prototype_bank.json`
- `outputs/cmcf/<exp>/prototype_mapping.jsonl` (mapping run 시)
- `outputs/cmcf/<exp>/viz/*_cmcf_parts_colored.ply`
- `outputs/cmcf/<exp>/viz/*_cmcf_part_boxes_lineset.ply`
- `outputs/cmcf/<exp>/viz/*_cmcf_markers.ply`

주의:
- `*_cmcf_parts_colored.ply` 색상은 CMCF marker-group box 기준(`part_color_source=cmcf_boxes`)
- `*_cmcf_part_boxes_lineset.ply` 박스는 CMCF marker-group box(`part_box_source=cmcf_marker_groups`)
- marker는 **magenta glyph**(작은 십자 점군)로 `*_cmcf_markers.ply`에 별도 저장됨
- GUI 토글 키:
  - `C`: part color ↔ raw.ply view
  - `B`: part box line on/off
  - `M`: marker on/off
