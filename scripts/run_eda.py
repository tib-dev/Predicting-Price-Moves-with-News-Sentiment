# scripts/run_eda.py
"""Run full EDA on financial news dataset."""

import argparse
import logging
import pandas as pd
from pathlib import Path

from fns_project.analysis.eda import run_full_eda

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def main(input_csv: str, output_dir: str):
    input_path = Path(input_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error("Input file does not exist: %s", input_path)
        return

    # Load dataset
    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows from %s", len(df), input_csv)

    # Run full EDA
    results = run_full_eda(df)

    # Save results to CSV
    for key, result_df in results.items():
        if isinstance(result_df, pd.DataFrame) and not result_df.empty:
            out_file = output_path / f"{key}.csv"
            result_df.to_csv(out_file, index=False)
            logger.info("Saved %s results to %s", key, out_file)

    logger.info("EDA pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EDA on news dataset")
    parser.add_argument(
        "--input_csv", type=str, required=True, help="Path to raw news CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/interim/eda_results",
        help="Directory to save EDA outputs",
    )
    args = parser.parse_args()
    main(args.input_csv, args.output_dir)
