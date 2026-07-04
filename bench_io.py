"""Stable benchmark output filenames and JSONL resume checkpoints."""
import json
import os


def benchmark_file_tag(model_type: str) -> str:
    """Short tag for result files, e.g. llava15_chair.jsonl."""
    return {
        "instructblip": "instructblip",
        "llava": "llava15",
        "llava-next": "llavanext",
    }.get(model_type, model_type.replace("-", "_"))


def load_jsonl_int_field(path: str, field: str) -> set:
    """Read JSONL; collect int(obj[field]) for each parseable line (skips bad lines)."""
    done = set()
    if not os.path.isfile(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)[field]))
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                continue
    return done
