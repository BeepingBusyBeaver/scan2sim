# scripts/pipeline/run_livehps_pipeline.py
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from scripts.common.io_paths import (
    DATA_INTEROP_EULER_DIR,
    DATA_INTEROP_LABEL_DIR,
    DATA_INTEROP_OBJ_DIR,
    DATA_INTEROP_QUAT_DIR,
    DATA_INTEROP_SMPL_DIR,
    DATA_INTERIM_DEBUG_EXTRACT_DIR,
    DATA_REAL_HUMAN_DIR,
    DATA_REAL_RAW_DIR,
    DATASET_LABEL_RULES_JSON,
    resolve_repo_path,
)


"""
python -m scripts.pipeline.run_livehps_pipeline \
  --pcap data/labeling/001/simpleMotion_wonjin.pcap \
  --base SN001 \
  --scan 861 893 929 961 989 \
         1032 1043 1073 1084 1122 \
         1132 1183 1205 1300 1401 \
         1488 1567 1596 1622 1655 \
         1675 \
  --raw-dir data/real/VAR/raw \
  --human-dir data/real/VAR/human \
  --smpl-dir outputs/VAR/smpl \
  --obj-dir outputs/VAR/obj \
  --quat-dir outputs/VAR/quat \
  --euler-dir outputs/VAR/euler \
  --label-dir outputs/VAR/label \
  --skip-pcap2pcd --skip-pcd2human


# have raw.pcd(bg+hm.pcd): --skip-pcap2pcd
# have human.ply: --skip-pcap2pcd --skip-pcd2human
# have smpl.npz: --skip-pcap2pcd --skip-pcd2human --skip-human2smpl
 
"""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def command_text(cmd: List[str]) -> str:
    return " ".join(cmd)


def run_step(cmd: List[str], dry_run: bool) -> None:
    print(f"$ {command_text(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unified LiveHPS pipeline: pcap2pcd -> pcd2human -> human2smpl -> npz2quat/npz2obj -> quat2unity -> unity2label."
    )
    parser.add_argument("--pcap", type=str, default="", help="Input raw pcap path.")
    parser.add_argument("--scan", type=int, nargs="+", default=None, help="Scan indices for pcap2pcd (e.g. --scan 0 10 20).")
    parser.add_argument("--base", type=str, default="SN001", help="Base prefix for exported raw pcd files.")

    parser.add_argument("--raw-dir", type=str, default=DATA_REAL_RAW_DIR, help="Raw PCD output directory.")
    parser.add_argument(
        "--raw-pattern",
        type=str,
        default=None,
        help="Input pattern for pcd2human. If omitted, uses '*.pcd' excluding background files.",
    )
    parser.add_argument(
        "--bg-pcd",
        type=str,
        default=None,
        help="Background pcd path for pcd2human. If omitted, tries '<raw-dir>/<base>_bg.pcd'.",
    )
    parser.add_argument("--no-bg", action="store_true", help="Run pcd2human without background subtraction input.")
    parser.add_argument("--pcd2human-max-range", type=float, default=4.0)
    parser.add_argument("--pcd2human-bg-thresh", type=float, default=0.05)
    parser.add_argument("--pcd2human-post-bg-thresh", type=float, default=0.0)
    parser.add_argument("--pcd2human-cluster-eps", type=float, default=0.012)
    parser.add_argument("--pcd2human-cluster-min-points", type=int, default=10)
    parser.add_argument("--pcd2human-person-xy-max", type=float, default=1.1)
    parser.add_argument("--pcd2human-person-area-max", type=float, default=0.9)
    parser.add_argument("--pcd2human-person-planarity-max", type=float, default=0.8)
    parser.add_argument("--pcd2human-vertical-plane-dist", type=float, default=0.01)
    parser.add_argument("--pcd2human-vertical-plane-min-inliers", type=int, default=300)
    parser.add_argument("--pcd2human-post-rad-radius", type=float, default=0.08)
    parser.add_argument("--pcd2human-post-rad-min-points", type=int, default=8)
    parser.add_argument("--pcd2human-bg-icp-voxel", type=float, default=0.06)
    parser.add_argument("--pcd2human-bg-icp-max-corr", type=float, default=0.12)
    parser.add_argument("--pcd2human-y-min", type=float, default=-0.9)
    parser.add_argument("--pcd2human-y-max", type=float, default=0.6)
    parser.add_argument("--pcd2human-debug-dir", type=str, default=DATA_INTERIM_DEBUG_EXTRACT_DIR)
    parser.add_argument("--pcd2human-bg-icp", dest="pcd2human_bg_icp", action="store_true")
    parser.add_argument("--no-pcd2human-bg-icp", dest="pcd2human_bg_icp", action="store_false")
    parser.add_argument("--pcd2human-apply-pcd-transform", dest="pcd2human_apply_pcd_transform", action="store_true")
    parser.add_argument("--no-pcd2human-apply-pcd-transform", dest="pcd2human_apply_pcd_transform", action="store_false")

    parser.add_argument("--human-dir", type=str, default=DATA_REAL_HUMAN_DIR, help="Human PLY output directory.")
    parser.add_argument("--human-pattern", type=str, default=None, help="Input pattern for human2smpl. default: current-run outputs only.")

    parser.add_argument("--smpl-dir", type=str, default=DATA_INTEROP_SMPL_DIR, help="SMPL npz output directory.")
    parser.add_argument("--quat-dir", type=str, default=DATA_INTEROP_QUAT_DIR, help="Quaternion json output directory.")
    parser.add_argument("--euler-dir", type=str, default=DATA_INTEROP_EULER_DIR, help="Unity euler json output directory.")
    parser.add_argument("--label-dir", type=str, default=DATA_INTEROP_LABEL_DIR, help="Label json output directory.")
    parser.add_argument("--obj-dir", type=str, default=DATA_INTEROP_OBJ_DIR, help="OBJ output directory.")
    parser.add_argument("--run-tag", type=str, default="", help="Run tag appended to intermediate output prefixes. default: no tag.")

    parser.add_argument("--weights", type=str, default="../LiveHPS/save_models/livehps.t7", help="LiveHPS weights path.")
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cuda", "cpu"], help="Inference device.")
    parser.add_argument("--num-points", type=int, default=256, help="Point samples per frame for human2smpl.")
    parser.add_argument("--smpl-pattern", type=str, default=None, help="Input pattern for npz2quat/npz2obj. default: current-run outputs only.")
    parser.add_argument("--quat-pattern", type=str, default=None, help="Input pattern for quat2unity. default: current-run outputs only.")
    parser.add_argument("--label-euler-pattern", type=str, default=None, help="Euler JSON pattern for unity2label. default: current-run outputs only.")
    parser.add_argument("--label-quat-pattern", type=str, default=None, help="Quaternion JSON pattern for unity2label. default: current-run outputs only.")
    parser.add_argument("--label-euler-prefix", type=str, default=None, help="Euler filename prefix hint for unity2label pairing.")
    parser.add_argument("--label-quat-prefix", type=str, default=None, help="Quaternion filename prefix hint for unity2label pairing.")

    parser.add_argument("--rules", type=str, default=DATASET_LABEL_RULES_JSON, help="Rules json for unity2label.")
    parser.add_argument("--label-frame", type=int, default=0, help="Frame index for unity2label.")
    parser.add_argument(
        "--label-joints",
        type=str,
        default=None,
        help="Comma-separated joints for unity2label. If omitted, uses classify_label.py default.",
    )

    parser.add_argument("--skip-pcap2pcd", action="store_true")
    parser.add_argument("--skip-pcd2human", action="store_true")
    parser.add_argument("--skip-human2smpl", action="store_true")
    parser.add_argument("--skip-post", action="store_true", help="Skip npz2quat/npz2obj/quat2unity/unity2label.")

    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable for subprocess steps.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.set_defaults(
        pcd2human_bg_icp=True,
        pcd2human_apply_pcd_transform=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()

    pcap_path = resolve_repo_path(root, args.pcap) if args.pcap else None
    raw_dir = resolve_repo_path(root, args.raw_dir)
    human_dir = resolve_repo_path(root, args.human_dir)
    smpl_dir = resolve_repo_path(root, args.smpl_dir)
    quat_dir = resolve_repo_path(root, args.quat_dir)
    euler_dir = resolve_repo_path(root, args.euler_dir)
    label_dir = resolve_repo_path(root, args.label_dir)
    obj_dir = resolve_repo_path(root, args.obj_dir)
    rules_json = resolve_repo_path(root, args.rules)
    run_tag = args.run_tag.strip()
    run_tag = "".join(char if (char.isalnum() or char in ("-", "_")) else "_" for char in run_tag)

    if run_tag:
        human_output_prefix = f"human_{run_tag}_"
        smpl_output_prefix = f"livehps_smpl_{run_tag}_"
        quat_output_prefix = f"livehps_quaternion_{run_tag}_"
        unity_output_prefix = f"livehps_unity_{run_tag}_"
        label_output_prefix = f"livehps_label_{run_tag}_"
    else:
        human_output_prefix = "human_"
        smpl_output_prefix = "livehps_smpl_"
        quat_output_prefix = "livehps_quaternion_"
        unity_output_prefix = "livehps_unity_"
        label_output_prefix = "livehps_label_"

    if args.human_pattern is not None:
        human_pattern = args.human_pattern
    elif args.skip_pcd2human:
        human_pattern = "*.ply"
    else:
        human_pattern = f"{human_output_prefix}*.ply"

    if args.smpl_pattern is not None:
        smpl_pattern = args.smpl_pattern
    elif args.skip_human2smpl:
        smpl_pattern = "*.npz"
    else:
        smpl_pattern = f"{smpl_output_prefix}*.npz"

    quat_pattern = args.quat_pattern or f"{quat_output_prefix}*.json"
    label_euler_pattern = args.label_euler_pattern or f"{unity_output_prefix}*.json"
    label_quat_pattern = args.label_quat_pattern or f"{quat_output_prefix}*.json"
    label_euler_prefix = args.label_euler_prefix if args.label_euler_prefix is not None else unity_output_prefix
    label_quat_prefix = args.label_quat_prefix if args.label_quat_prefix is not None else quat_output_prefix

    default_bg_pcd = raw_dir / f"{args.base}_bg.pcd"
    if args.bg_pcd:
        bg_pcd = resolve_repo_path(root, args.bg_pcd)
    elif default_bg_pcd.exists():
        bg_pcd = default_bg_pcd
    else:
        bg_pcd = None

    steps: List[List[str]] = []
    scan_list = [int(s) for s in (args.scan or [])]

    if not args.skip_pcap2pcd:
        if pcap_path is None:
            raise ValueError("--pcap is required unless --skip-pcap2pcd is set.")
        if not scan_list:
            raise ValueError("--scan is required unless --skip-pcap2pcd is set.")

    raw_inputs: List[str]
    if args.raw_pattern:
        raw_inputs = [str(raw_dir / args.raw_pattern)]
    elif not args.skip_pcap2pcd and scan_list:
        raw_inputs = [str(raw_dir / f"{args.base}_{scan_idx:06d}.pcd") for scan_idx in scan_list]
    else:
        bg_candidates = set()
        if bg_pcd is not None:
            bg_candidates.add(bg_pcd.resolve())
        raw_candidates = sorted(raw_dir.glob("*.pcd"))
        filtered = [
            path.resolve()
            for path in raw_candidates
            if path.is_file()
            and path.suffix.lower() == ".pcd"
            and path.resolve() not in bg_candidates
            and not path.stem.lower().endswith("_bg")
        ]
        raw_inputs = [str(path) for path in filtered] if filtered else [str(raw_dir / "*.pcd")]

    if not args.skip_pcap2pcd:
        steps.append(
            [
                args.python,
                "-m",
                "scripts",
                "pcap2pcd",
                str(pcap_path),
                "--scan",
                *[str(s) for s in scan_list],
                "--outdir",
                str(raw_dir),
                "--base",
                args.base,
            ]
        )

    if not args.skip_pcd2human:
        cmd = [
            args.python,
            "-m",
            "scripts",
            "pcd2human",
            "--in",
            *raw_inputs,
            "--out_dir",
            str(human_dir),
            "--output_mode",
            "human_index",
            "--output_prefix",
            human_output_prefix,
            "--output_start",
            "1",
            "--output_digits",
            "3",
            "--max_range",
            str(float(args.pcd2human_max_range)),
            "--bg_thresh",
            str(float(args.pcd2human_bg_thresh)),
            "--post_bg_thresh",
            str(float(args.pcd2human_post_bg_thresh)),
            "--cluster_eps",
            str(float(args.pcd2human_cluster_eps)),
            "--cluster_min_points",
            str(int(args.pcd2human_cluster_min_points)),
            "--person_xy_max",
            str(float(args.pcd2human_person_xy_max)),
            "--person_area_max",
            str(float(args.pcd2human_person_area_max)),
            "--person_planarity_max",
            str(float(args.pcd2human_person_planarity_max)),
            "--vertical_plane_dist",
            str(float(args.pcd2human_vertical_plane_dist)),
            "--vertical_plane_min_inliers",
            str(int(args.pcd2human_vertical_plane_min_inliers)),
            "--post_rad_radius",
            str(float(args.pcd2human_post_rad_radius)),
            "--post_rad_min_points",
            str(int(args.pcd2human_post_rad_min_points)),
            "--bg_icp_voxel",
            str(float(args.pcd2human_bg_icp_voxel)),
            "--bg_icp_max_corr",
            str(float(args.pcd2human_bg_icp_max_corr)),
            "--y_min",
            str(float(args.pcd2human_y_min)),
            "--y_max",
            str(float(args.pcd2human_y_max)),
        ]
        if (not args.no_bg) and (bg_pcd is not None):
            cmd.extend(["--bg", str(bg_pcd)])
        if args.pcd2human_bg_icp:
            cmd.append("--bg_icp")
        if not args.pcd2human_apply_pcd_transform:
            cmd.append("--no_pcd_transform")
        if args.pcd2human_debug_dir:
            cmd.extend(["--debug_dir", str(resolve_repo_path(root, args.pcd2human_debug_dir))])
        steps.append(cmd)

    if not args.skip_human2smpl:
        steps.append(
            [
                args.python,
                "-m",
                "scripts",
                "human2smpl",
                "--input",
                str(human_dir),
                "--batch_per_file",
                "--input_pattern",
                human_pattern,
                "--output_dir",
                str(smpl_dir),
                "--output_prefix",
                smpl_output_prefix,
                "--weights",
                args.weights,
                "--device",
                args.device,
                "--num_points",
                str(int(args.num_points)),
            ]
        )

    if not args.skip_post:
        label_cmd = [
            args.python,
            "-m",
            "scripts",
            "unity2label",
            "--batch_per_file",
            "--euler-json",
            str(euler_dir),
            "--quat-json",
            str(quat_dir),
            "--euler-pattern",
            label_euler_pattern,
            "--quat-pattern",
            label_quat_pattern,
            "--euler-prefix",
            label_euler_prefix,
            "--quat-prefix",
            label_quat_prefix,
            "--rules",
            str(rules_json),
            "--output-dir",
            str(label_dir),
            "--output-prefix",
            label_output_prefix,
            "--frame",
            str(int(args.label_frame)),
        ]
        if args.label_joints:
            label_cmd.extend(["--joints", args.label_joints])

        steps.extend(
            [
                [
                    args.python,
                    "-m",
                    "scripts",
                    "npz2quat",
                    "--input",
                    str(smpl_dir),
                    "--batch_per_file",
                    "--input_pattern",
                    smpl_pattern,
                    "--output_dir",
                    str(quat_dir),
                    "--output_prefix",
                    quat_output_prefix,
                ],
                [
                    args.python,
                    "-m",
                    "scripts",
                    "npz2obj",
                    "--input",
                    str(smpl_dir / smpl_pattern),
                    "--output",
                    str(obj_dir),
                ],
                [
                    args.python,
                    "-m",
                    "scripts",
                    "quat2unity",
                    "--input",
                    str(quat_dir),
                    "--batch_per_file",
                    "--input_pattern",
                    quat_pattern,
                    "--output_dir",
                    str(euler_dir),
                    "--output_prefix",
                    unity_output_prefix,
                ],
                label_cmd,
            ]
        )

    print(f"[pipeline] root={root}")
    if run_tag:
        print(f"[pipeline] run_tag={run_tag}")
    for cmd in steps:
        run_step(cmd, dry_run=bool(args.dry_run))
    print("[pipeline] done")


if __name__ == "__main__":
    main()
