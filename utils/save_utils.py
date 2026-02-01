import os
import json
from pathlib import Path


def atomic_json_dump(obj, out_path, indent=2):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    tmp.replace(out_path)

def load_results(exp_save_path):
    if not os.path.exists(exp_save_path):
        return {}
    with open(exp_save_path, 'r', encoding='utf-8') as f:
        return json.load(f)
