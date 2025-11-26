"""
Quick notebook setup:
- Adds src/ to sys.path
- Loads config (optional experiment)
- Initializes logger
- Sets key paths and variables
"""

from ..logging_config import setup_logger
from ..config import ConfigLoader
from pathlib import Path
import sys
import configparser

# -----------------------------
# Add src/ and project root to Python path
# -----------------------------
project_root = Path(__file__).parents[3]  # utils -> fns_project -> src -> root
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))

# -----------------------------
# Helper to safely read sections from cfg
# -----------------------------


def _read_section(cfg, section_name):
    """
    Return a plain dict for the requested section.
    Handles ConfigParser-like objects and dict-like objects.
    If the section is missing, returns an empty dict.
    """
    # ConfigParser (ini-style)
    if isinstance(cfg, configparser.ConfigParser):
        if cfg.has_section(section_name):
            return dict(cfg.items(section_name))
        return {}

    # dict-like / other objects exposing .get()
    try:
        section = cfg.get(section_name)
    except Exception:
        section = None

    # If cfg.get returned a dict-like section
    if isinstance(section, dict):
        return section

    # Fallback: try indexing like cfg[section_name]
    try:
        maybe = cfg[section_name]
        if isinstance(maybe, dict):
            return maybe
    except Exception:
        pass

    # Not found -> return empty dict
    return {}


# -----------------------------
# Load config & initialize logger
# -----------------------------
cfg = ConfigLoader(experiment_file=None)  # or experiment_X.yaml
logger = setup_logger(cfg, name="notebook")

# -----------------------------
# Paths (use safe lookups + sensible defaults)
# -----------------------------
paths = _read_section(cfg, "paths")
output = _read_section(cfg, "output")
data = _read_section(cfg, "data")

RAW_DIR = Path(paths.get("raw_data", "raw"))
INTERIM_DIR = Path(paths.get("interim_data", "interim"))
PROCESSED_DIR = Path(paths.get("processed_data", "processed"))
FEATURE_DIR = Path(paths.get("feature_store", "features"))
PLOTS_DIR = Path(output.get("plots_dir", "outputs/plots"))

# -----------------------------
# Assets & date range (use defaults if missing)
# -----------------------------
ASSETS = data.get("assets", [])
START_DATE = data.get("start_date", None)
END_DATE = data.get("end_date", None)

logger.info(
    f"Notebook initialized: assets={ASSETS}, date range={START_DATE} → {END_DATE}"
)
