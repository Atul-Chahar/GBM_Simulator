# Contributing

Guidelines for contributing to the BTC GBM Forecast project.

---

## Setup

```bash
# Clone and create virtual environment
git clone <repo-url>
cd GBM_simulator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `model/` | Core prediction logic (data fetching, GBM engine, evaluation) |
| `persistence/` | Prediction storage backends |
| `static/` | CSS design system |
| `.streamlit/` | Streamlit server config |

## Development Workflow

### Running Locally

```bash
# Dashboard
streamlit run app.py

# Backtest (takes ~25 min)
python backtest.py

# Quick validation (200 bars, ~5 min)
python -c "
from backtest import run_backtest
run_backtest(test_bars=200, output_file='validation.jsonl')
"
```

### Making Changes to the Model

1. **Always validate** changes with a short backtest (200 bars) before running the full 720-bar test
2. **Target 95.0% coverage** — not higher, not lower. The Winkler score penalizes both misses AND unnecessarily wide intervals
3. **Key tuning parameters** in `gbm_engine.py`:
   - `_cal_factor` (0.85): Variance calibration. Lower = narrower intervals
   - `alpha` (0.15): Crisis detection multiplier
   - `delta` (0.10): Baseline redundancy
   - `redundancy` (0.02): Minimum floor

### Making Changes to the Dashboard

1. **CSS**: Edit `static/styles.css` — follows the xAI design system in `design.md`
2. **Layout**: Edit `app.py` — uses Streamlit + custom HTML
3. **Theme**: Both dark and light themes must be tested
4. **Charts**: Plotly with theme-aware colors via `theme_colors()` helper

## Code Style

- **Python**: Standard PEP 8, type hints where practical
- **Docstrings**: Google style for all public functions
- **Comments**: Explain *why*, not *what*
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes

## Testing Checklist

Before submitting changes:

- [ ] `python backtest.py` produces 720 predictions
- [ ] Coverage is between 93% and 97%
- [ ] `streamlit run app.py` loads without errors
- [ ] Both dark and light themes render correctly
- [ ] All 6 chart panels display data
- [ ] Prediction history saves and displays

## Key Files Not to Break

| File | Why |
|------|-----|
| `gbm_engine.py` | Core model — any change affects coverage |
| `evaluator.py` | Scoring must match challenge specification exactly |
| `backtest_results.jsonl` | Required submission artifact |
| `requirements.txt` | Deployment depends on exact packages |

## Deployment

### Streamlit Community Cloud

1. Push to GitHub (public or private repo)
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Set main file to `app.py`
4. (Optional) Add secrets for Google Sheets persistence

### Environment Variables / Secrets

For Google Sheets persistence (Part C), add to `.streamlit/secrets.toml`:

```toml
[gcp_service_account]
type = "service_account"
project_id = "your-project"
private_key = "-----BEGIN PRIVATE KEY-----\n..."
client_email = "your-sa@your-project.iam.gserviceaccount.com"
```

## License

MIT
