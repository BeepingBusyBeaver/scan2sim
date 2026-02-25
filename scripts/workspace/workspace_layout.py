# scripts/workspace/workspace_layout.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from scripts.common.json_utils import read_json, write_json

DEFAULT_CONFIG = {
    "livehps_root": "../LiveHPS",
    "scan2sim_root": ".",
    "livehps_reference_dirs": [
        "Model",
        "dataset",
        "save_models",
        "smpl",
    ],
    "scan2sim_contract_dirs": [
        "configs",
        "dataset",
        "docs/integration",
        "scripts/common",
        "scripts/conversion",
        "scripts/labeling",
        "scripts/infer",
        "scripts/pipeline",
        "scripts/workspace",
        "data/real/raw",
        "data/real/human",
        "data/real/VAR",
        "outputs/smpl",
        "outputs/quat",
        "outputs/euler",
        "outputs/label",
        "outputs/obj",
        "data/match_one",
        "integrations/livehps/overrides",
        "logs",
    ],
    "scan2sim_recommended_cleanup": {
        "prefer_outputs_interop_over_data_interop": True,
        "prefer_data_match_one_over_outputs": True,
        "prefer_data_over_output_data_clones": True,
    },
}


@dataclass
class LayoutReport:
    root_label: str
    created: int
    existed: int
    missing_root: bool


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_path_from_args(config_arg: str) -> Path:
    path = Path(config_arg).expanduser()
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def merge_with_default_config(user_cfg: Dict) -> Dict:
    merged = dict(DEFAULT_CONFIG)
    for key in ("livehps_reference_dirs", "scan2sim_contract_dirs"):
        merged[key] = list(DEFAULT_CONFIG[key])
    if isinstance(user_cfg, dict):
        for key, value in user_cfg.items():
            merged[key] = value
        if "livehps_contract_dirs" in user_cfg and "livehps_reference_dirs" not in user_cfg:
            merged["livehps_reference_dirs"] = list(user_cfg["livehps_contract_dirs"])
    return merged


def resolve_roots(config_obj: Dict) -> Tuple[Path, Path]:
    root = repo_root()
    livehps_root = (root / str(config_obj.get("livehps_root", "../LiveHPS"))).resolve()
    scan2sim_root = (root / str(config_obj.get("scan2sim_root", "."))).resolve()
    return livehps_root, scan2sim_root


def create_default_config(config_path: Path, force: bool) -> None:
    if config_path.exists() and not force:
        print(f"[SKIP] config exists: {config_path}")
        print("       use --force to overwrite.")
        return
    write_json(config_path, DEFAULT_CONFIG, ascii_flag=False, compact=False)
    print(f"[OK] config written: {config_path}")


def ensure_config(config_path: Path) -> Dict:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            "Run `python -m scripts workspace-init` first."
        )
    user_cfg = read_json(config_path)
    if not isinstance(user_cfg, dict):
        raise ValueError(f"Invalid config json (must be object): {config_path}")
    return merge_with_default_config(user_cfg)


def collect_existing_dirs(root: Path, relative_dirs: List[str]) -> Tuple[int, int]:
    exists = 0
    missing = 0
    for rel in relative_dirs:
        full = (root / rel).resolve()
        if full.exists() and full.is_dir():
            exists += 1
        else:
            missing += 1
    return exists, missing


def detect_scan2sim_duplications(scan2sim_root: Path) -> List[str]:
    findings: List[str] = []
    data_interop = (scan2sim_root / "data" / "interop").resolve()
    outputs_interop_dirs = [
        (scan2sim_root / "outputs" / "smpl").resolve(),
        (scan2sim_root / "outputs" / "quat").resolve(),
        (scan2sim_root / "outputs" / "euler").resolve(),
        (scan2sim_root / "outputs" / "label").resolve(),
        (scan2sim_root / "outputs" / "obj").resolve(),
    ]
    if data_interop.exists() and any(path.exists() for path in outputs_interop_dirs):
        findings.append("both `data/interop/*` and `outputs/*` interop dirs exist; consolidate to `outputs/*`.")
    elif data_interop.exists():
        findings.append("legacy `data/interop/*` exists; migrate to `outputs/*`.")

    outputs_match_one = (scan2sim_root / "outputs" / "match_one").resolve()
    data_match_one = (scan2sim_root / "data" / "match_one").resolve()
    if outputs_match_one.exists() and data_match_one.exists():
        findings.append("both `outputs/match_one` and `data/match_one` exist; consolidate to `data/match_one`.")
    elif outputs_match_one.exists():
        findings.append("legacy `outputs/match_one` exists; migrate to `data/match_one`.")

    output_interim = (scan2sim_root / "output" / "interim").resolve()
    data_interim = (scan2sim_root / "data" / "interim").resolve()
    if output_interim.exists() and data_interim.exists():
        findings.append("both `output/interim` and `data/interim` exist; avoid duplicate lifecycle paths.")

    output_real_raw = (scan2sim_root / "output" / "real" / "raw").resolve()
    data_real_raw = (scan2sim_root / "data" / "real" / "raw").resolve()
    if output_real_raw.exists() and data_real_raw.exists():
        findings.append("both `output/real/raw` and `data/real/raw` exist; consolidate ownership.")

    return findings


def print_doctor_summary(config_obj: Dict) -> None:
    livehps_root, scan2sim_root = resolve_roots(config_obj)
    print("[Workspace Layout Doctor]")
    print(f"- livehps_root: {livehps_root} (exists={livehps_root.exists()})")
    print(f"- scan2sim_root: {scan2sim_root} (exists={scan2sim_root.exists()})")

    live_exists, live_missing = collect_existing_dirs(livehps_root, list(config_obj["livehps_reference_dirs"]))
    print(
        f"- livehps reference dirs: total={len(config_obj['livehps_reference_dirs'])}, "
        f"exists={live_exists}, missing={live_missing}"
    )

    scan_exists, scan_missing = collect_existing_dirs(scan2sim_root, list(config_obj["scan2sim_contract_dirs"]))
    print(
        f"- scan2sim contract dirs: total={len(config_obj['scan2sim_contract_dirs'])}, "
        f"exists={scan_exists}, missing={scan_missing}"
    )

    duplicates = detect_scan2sim_duplications(scan2sim_root)
    if duplicates:
        print("- scan2sim duplication findings:")
        for item in duplicates:
            print(f"  - {item}")
    else:
        print("- scan2sim duplication findings: none")


def ensure_dir(path: Path, dry_run: bool) -> bool:
    if path.exists():
        return False
    if dry_run:
        return True
    path.mkdir(parents=True, exist_ok=True)
    return True


def apply_layout(config_obj: Dict, dry_run: bool) -> List[LayoutReport]:
    livehps_root, scan2sim_root = resolve_roots(config_obj)
    reports: List[LayoutReport] = []

    if not livehps_root.exists():
        reports.append(LayoutReport(root_label="livehps-ref", created=0, existed=0, missing_root=True))
    else:
        exists, _missing = collect_existing_dirs(livehps_root, list(config_obj["livehps_reference_dirs"]))
        reports.append(LayoutReport(root_label="livehps-ref", created=0, existed=exists, missing_root=False))

    if not scan2sim_root.exists():
        reports.append(LayoutReport(root_label="scan2sim", created=0, existed=0, missing_root=True))
        return reports

    created = 0
    existed = 0
    for rel in list(config_obj["scan2sim_contract_dirs"]):
        full = (scan2sim_root / rel).resolve()
        if full.exists():
            existed += 1
            continue
        if ensure_dir(full, dry_run=dry_run):
            created += 1

    reports.append(LayoutReport(root_label="scan2sim", created=created, existed=existed, missing_root=False))

    return reports


def print_layout_summary(reports: List[LayoutReport], dry_run: bool) -> None:
    print(f"[Workspace Layout Apply] dry_run={dry_run}")
    for report in reports:
        if report.missing_root:
            print(f"- {report.root_label}: root missing (skipped)")
            continue
        print(f"- {report.root_label}: created={report.created}, already_exists={report.existed}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Folder layout manager for scan2sim + LiveHPS reference workspace."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/workspace_layout.json",
        help="Config path (JSON). Relative paths are resolved from scan2sim root.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create default layout config.")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing config.")

    subparsers.add_parser("doctor", help="Check folder-layout health and duplication hints.")

    layout_parser = subparsers.add_parser("layout", help="Create missing contract directories only.")
    layout_parser.add_argument("--dry-run", action="store_true", help="Print actions only, do not write.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_path = config_path_from_args(args.config)

    if args.command == "init":
        create_default_config(config_path=config_path, force=args.force)
        return

    config_obj = ensure_config(config_path)

    if args.command == "doctor":
        print_doctor_summary(config_obj)
        return

    if args.command == "layout":
        reports = apply_layout(config_obj=config_obj, dry_run=args.dry_run)
        print_layout_summary(reports=reports, dry_run=args.dry_run)
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
