# scripts/data/rename_virtual.py
from __future__ import annotations

import argparse
import re
from pathlib import Path

from scripts.common.io_paths import DATA_VIRTUAL_POSE324_DIR, resolve_repo_path

"""
python -m scripts.data.rename_virtual \
    "data/virtual/pose324"
"""

PATTERN = re.compile(r"^virtual_(\d+)\.ply$")

def compute_new_name(n: int, views: int, sep: str) -> str:
    a = n % views
    b = n // views
    return f"virtual_{a}{sep}{b}.ply"

def main() -> None:
    parser = argparse.ArgumentParser(description="Rename virtual_###.ply files in a directory.")
    parser.add_argument("dir", nargs="?", default=DATA_VIRTUAL_POSE324_DIR, help="대상 폴더 경로")
    parser.add_argument("--views", type=int, default=1000,
                        help="a=n%%views, b=n//views (기본 1000이면 000~999는 b=0)")
    parser.add_argument("--sep", default="-",
                        help="두 숫자 사이 구분자 (예: '-' 또는 '_')")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 변경 없이 변경 목록만 출력")
    parser.add_argument("--overwrite", action="store_true",
                        help="대상 파일이 이미 있으면 덮어쓰기 (주의)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    base = resolve_repo_path(repo_root, args.dir)
    if not base.is_dir():
        raise SystemExit(f"폴더가 아닙니다: {base}")

    files = sorted(base.glob("virtual_*.ply"))
    if not files:
        print("매칭되는 파일이 없습니다.")
        return

    planned: list[tuple[Path, Path]] = []
    for p in files:
        m = PATTERN.match(p.name)
        if not m:
            continue
        n = int(m.group(1))
        new_name = compute_new_name(n, args.views, args.sep)
        new_path = p.with_name(new_name)
        planned.append((p, new_path))

    if not planned:
        print("규칙(virtual_숫자.ply)에 맞는 파일이 없습니다.")
        return

    # 충돌 검사
    for src, dst in planned:
        if dst.exists() and not args.overwrite:
            raise SystemExit(f"대상 파일이 이미 존재합니다(중단): {dst}\n"
                             f"덮어쓰려면 --overwrite 사용")

    # 실행
    for src, dst in planned:
        if args.dry_run:
            print(f"[DRY] {src.name} -> {dst.name}")
        else:
            if dst.exists() and args.overwrite:
                # rename은 OS에 따라 기존 파일을 덮어쓸 수 있어, 명시적으로 제거 후 rename
                dst.unlink()
            src.rename(dst)  # rename 동작: Path.rename() 사용 :contentReference[oaicite:1]{index=1}
            print(f"{src.name} -> {dst.name}")

if __name__ == "__main__":
    main()
