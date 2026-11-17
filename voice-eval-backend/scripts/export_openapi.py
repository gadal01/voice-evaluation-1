"""Writes the API's OpenAPI document (no server, no AWS) for the voice-eval-ui repo to generate its types from.

    python scripts/export_openapi.py ../voice-eval-ui/openapi.json
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("AUTH_API_KEY", "x" * 20)

from voice_eval.app import create_app  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "openapi.json")
out.write_text(json.dumps(create_app().openapi(), indent=2), encoding="utf-8")
print("wrote", out.resolve())
