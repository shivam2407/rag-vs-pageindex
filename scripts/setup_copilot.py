# Re-export from 00_setup_copilot for backward-compatible imports
import importlib.util as _ilu
from pathlib import Path as _P

_spec = _ilu.spec_from_file_location(
    "_setup", str(_P(__file__).parent / "00_setup_copilot.py"))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

get_client = _mod.get_client
get_copilot_token = _mod.get_copilot_token
GITHUB_MODELS_URL = _mod.GITHUB_MODELS_URL
