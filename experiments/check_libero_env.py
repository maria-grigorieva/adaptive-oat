#!/usr/bin/env python
"""
Lightweight dependency probe for optional LIBERO smoke evaluation.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Dict


DEPENDENCIES = ("mujoco", "robosuite", "libero")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check optional LIBERO smoke-eval dependencies.")
    parser.add_argument("--output-dir", default="results/libero_smoke")
    return parser.parse_args()


def probe_import(module_name: str) -> Dict[str, object]:
    try:
        module = importlib.import_module(module_name)
        return {
            "available": True,
            "error": None,
            "module": module_name,
            "version": getattr(module, "__version__", None),
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "module": module_name,
            "version": None,
        }


def format_cell(value: str, width: int) -> str:
    return value.ljust(width)


def print_diagnostic_table(report: Dict[str, Dict[str, object]]) -> None:
    headers = ("dependency", "available", "version", "detail")
    rows = []
    for module_name in DEPENDENCIES:
        details = report[module_name]
        rows.append(
            (
                module_name,
                "yes" if details["available"] else "no",
                str(details["version"] or "-"),
                str(details["error"] or "import ok"),
            )
        )

    widths = [
        max(len(headers[idx]), max(len(str(row[idx])) for row in rows))
        for idx in range(len(headers))
    ]
    header_line = " | ".join(format_cell(headers[idx], widths[idx]) for idx in range(len(headers)))
    separator = "-+-".join("-" * widths[idx] for idx in range(len(headers)))
    print(header_line)
    print(separator)
    for row in rows:
        print(" | ".join(format_cell(str(row[idx]), widths[idx]) for idx in range(len(headers))))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {module_name: probe_import(module_name) for module_name in DEPENDENCIES}
    all_available = all(bool(details["available"]) for details in report.values())
    payload = {
        "all_available": all_available,
        "dependencies": report,
    }

    output_path = output_dir / "libero_env_check.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print_diagnostic_table(report)
    print(f"\nWrote diagnostics to {output_path}")


if __name__ == "__main__":
    main()
