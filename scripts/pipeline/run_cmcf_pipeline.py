# scripts/pipeline/run_cmcf_pipeline.py
from __future__ import annotations

from typing import Sequence

from scripts.cmcf.run_cmcf_pipeline import main as cmcf_main


def main(argv: Sequence[str] | None = None) -> None:
    cmcf_main(list(argv) if argv is not None else None)


if __name__ == "__main__":
    main()
