"""Verify that `lean/LabEarsLean.lean` builds cleanly and uses only standard Lean 4 axioms.

Checks:
  1. Zero `sorry` tokens or custom `axiom` declarations in `lean/LabEarsLean.lean`.
  2. Every `theorem` in `LabEarsLean` compiles under `lake env lean` with `#print axioms`
     reporting either `'...' does not depend on any axioms` or a subset of
     `{propext, Classical.choice, Quot.sound}`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEAN_DIR = REPO_ROOT / "lean"
LEAN_FILE = LEAN_DIR / "LabEarsLean.lean"
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def _find_lake() -> str:
    env_lake = os.environ.get("LAKE_BIN")
    if env_lake and Path(env_lake).is_file():
        return env_lake
    which_lake = shutil.which("lake")
    if which_lake:
        return which_lake
    win_elan = Path.home() / ".elan" / "bin" / ("lake.exe" if os.name == "nt" else "lake")
    if win_elan.is_file():
        return str(win_elan)
    raise FileNotFoundError("Could not locate `lake` executable on PATH or ~/.elan/bin")


def _strip_comments(source: str) -> str:
    no_block = re.sub(r"/-.*?-/", "", source, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", no_block)


def main() -> int:
    if not LEAN_FILE.is_file():
        print(f"error: missing {LEAN_FILE}", file=sys.stderr)
        return 1

    raw = LEAN_FILE.read_text(encoding="utf-8")
    code = _strip_comments(raw)

    if re.search(r"\bsorry\b", code):
        print("error: `sorry` detected in LabEarsLean.lean", file=sys.stderr)
        return 1
    if re.search(r"^\s*axiom\s+", code, flags=re.MULTILINE):
        print("error: custom `axiom` declaration detected in LabEarsLean.lean", file=sys.stderr)
        return 1

    theorems = re.findall(r"\btheorem\s+([A-Za-z0-9_']+)", code)
    if not theorems:
        print("error: zero theorems discovered in LabEarsLean.lean", file=sys.stderr)
        return 1

    lake_bin = _find_lake()
    build_proc = subprocess.run([lake_bin, "build"], cwd=LEAN_DIR, capture_output=True, text=True)
    if build_proc.returncode != 0:
        sys.stderr.write(build_proc.stdout + build_proc.stderr)
        return build_proc.returncode

    check_lines = ["import LabEarsLean"] + [f"#print axioms LabEarsLean.{thm}" for thm in theorems]
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".lean",
        dir=LEAN_DIR,
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write("\n".join(check_lines) + "\n")
        tmp_path = Path(tmp.name)

    try:
        ax_proc = subprocess.run(
            [lake_bin, "env", "lean", str(tmp_path)],
            cwd=LEAN_DIR,
            capture_output=True,
            text=True,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    if ax_proc.returncode != 0:
        sys.stderr.write(ax_proc.stdout + ax_proc.stderr)
        return ax_proc.returncode

    output = (ax_proc.stdout + "\n" + ax_proc.stderr).strip()
    seen_theorems: set[str] = set()
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        m_none = re.fullmatch(r"'LabEarsLean\.([A-Za-z0-9_']+)' does not depend on any axioms", line)
        if m_none:
            seen_theorems.add(m_none.group(1))
            continue
        m_dep = re.fullmatch(r"'LabEarsLean\.([A-Za-z0-9_']+)' depends on axioms: \[(.*)\]", line)
        if m_dep:
            seen_theorems.add(m_dep.group(1))
            used = {tok.strip() for tok in m_dep.group(2).split(",") if tok.strip()}
            disallowed = used - ALLOWED_AXIOMS
            if disallowed:
                print(
                    f"error: theorem {m_dep.group(1)} uses disallowed axioms: {sorted(disallowed)}",
                    file=sys.stderr,
                )
                return 1
            continue
        print(f"error: unexpected Lean output line: {line}", file=sys.stderr)
        return 1

    missing = set(theorems) - seen_theorems
    if missing:
        print(f"error: missing #print axioms output for: {sorted(missing)}", file=sys.stderr)
        return 1

    print(f"Verified {len(theorems)} LabEarsLean theorems (zero sorry, standard axioms only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
