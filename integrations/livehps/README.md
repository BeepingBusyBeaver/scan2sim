# LiveHPS Integration Layout

This folder is now a lightweight traceability area for upstream override snapshots.

## Unified structure
scan2sim-level integration paths are:
- `scripts/`: LiveHPS-related utilities are merged into common script groups (`conversion`, `labeling`, `infer`, `workspace`)
- `outputs/`: runtime inputs/outputs (`smpl`, `quat`, `euler`, `label`, `obj`)
- `docs/integration/`: integration notes (e.g., setup memo)
- `integrations/livehps/overrides/`: upstream-file override snapshots only
