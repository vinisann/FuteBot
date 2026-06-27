import os


def get_api_key(st_module=None):
    """Read the optional Football-Data API key without falling back to a real token."""
    env_key = os.environ.get("FOOTBALL_DATA_API_KEY", "").strip()
    if env_key:
        return env_key

    if st_module is None:
        return ""

    try:
        return str(st_module.secrets.get("FOOTBALL_DATA_API_KEY", "")).strip()
    except Exception:
        return ""
