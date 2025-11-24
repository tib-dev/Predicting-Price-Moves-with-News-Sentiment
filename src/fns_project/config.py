import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigLoader:
    def __init__(self, config_dir: str = "configs", experiment_file: Optional[str] = None):
        """
        Load default config and optionally an experiment override.

        Args:
            config_dir (str): Path to the configs directory.
            experiment_file (str, optional): Filename of experiment YAML to override defaults.
        """
        self.config_dir = Path(config_dir)
        self.default_config = self._load_yaml(self.config_dir / "default.yaml")
        self.indicators_config = self._load_yaml(
            self.config_dir / "indicators.yaml")
        self.sentiment_config = self._load_yaml(
            self.config_dir / "sentiment.yaml")
        self.experiment_config = self._load_yaml(
            self.config_dir / experiment_file) if experiment_file else {}

        # Merge configs
        self.config = self._merge_configs(
            self.default_config, self.experiment_config)

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge override into base.
        """
        result = base.copy()
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = self._merge_configs(result[k], v)
            else:
                result[k] = v
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """
        Access top-level config keys
        """
        return self.config.get(key, default)

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Access nested sections like 'indicators_config' or 'sentiment_config'
        """
        if section == "indicators":
            return self.indicators_config
        elif section == "sentiment":
            return self.sentiment_config
        return self.config.get(section, {})


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Load default + optional experiment
    cfg = ConfigLoader(experiment_file="experiment_X.yaml")

    print("Project Name:", cfg.get("project_name"))
    print("Assets:", cfg.get("data")["assets"])
    print("Indicators:", cfg.get_section("indicators")["moving_averages"])
    print("Sentiment model:", cfg.get_section(
        "sentiment")["sentiment"]["model"])
