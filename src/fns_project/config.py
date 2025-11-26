import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigLoader:
    """
    Unified config loader with experiment overrides,
    automatic path creation, and easy section access.
    """

    def __init__(
        self,
        config_dir: Optional[str] = None,
        experiment_file: Optional[str] = None
    ):
        # Determine correct config directory
        if config_dir is None:
            project_root = Path(__file__).parents[2]
            config_dir = project_root / "configs"

        self.config_dir = Path(config_dir)

        # Load base configs
        self.default_config = self._load_yaml(self.config_dir / "default.yaml")
        self.indicators_config = self._load_yaml(
            self.config_dir / "indicators.yaml")
        self.sentiment_config = self._load_yaml(
            self.config_dir / "sentiment.yaml")

        # Optional experiment override
        self.experiment_config = (
            self._load_yaml(self.config_dir / experiment_file)
            if experiment_file
            else {}
        )

        # Merge default + experiment
        self.config = self._merge_configs(
            self.default_config, self.experiment_config)

        # Attach other config blocks
        self.config["indicators"] = self.indicators_config
        self.config["sentiment"] = self.sentiment_config

        # Create required directories
        self._ensure_paths_exist()

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = base.copy()
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = self._merge_configs(result[k], v)
            else:
                result[k] = v
        return result

    def _ensure_paths_exist(self):
        for p in self.config.get("paths", {}).values():
            Path(p).mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def get_section(self, section: str) -> Dict[str, Any]:
        return self.config.get(section, {})
