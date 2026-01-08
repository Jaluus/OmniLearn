"""CLI bootstrap to make `omnilearn` importable without installation.

This keeps the existing workflow:
  cd OmniLearn/scripts && python train.py ...
while allowing imports like `from omnilearn.models.pet import PET`.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_project_root_on_path() -> None:
    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parent  # OmniLearn/
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


_ensure_project_root_on_path()

