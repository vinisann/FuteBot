# FuteBot

FuteBot is a Streamlit app for World Cup match tracking, offline fallback data, Poisson/ELO predictions, tournament simulations, historical stats, and prediction accuracy reports.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install runtime dependencies:

```powershell
pip install -r requirements.txt
```

For local tests, install development dependencies:

```powershell
pip install -r requirements-dev.txt
```

## Optional API Token

The app works in offline fallback mode without a token. To sync with Football-Data.org, set your token with one of these options:

```powershell
$env:FOOTBALL_DATA_API_KEY="your-token-here"
```

Or copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml` and set:

```toml
FOOTBALL_DATA_API_KEY = "your-token-here"
```

Never commit `.streamlit/secrets.toml`.

## Run

```powershell
streamlit run app.py
```

The SQLite database is generated locally under `data/` on first run. Local `.db` files are intentionally ignored by Git.

## QA Checks

```powershell
$env:PYTHONIOENCODING="utf-8"
python -m compileall app.py src pages
python -m compileall scratch
pytest -q
```

## Notes

- Embedded 2026 data is treated as offline fallback seed data.
- Prediction accuracy excludes 2026 fallback seed matches and predicts each match using only earlier historical data.
- API-synced matches are marked separately from local seed data in the database.
