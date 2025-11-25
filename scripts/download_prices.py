

import argparse
from pathlib import Path
from fns_project.data.loader import load_csv
from fns_project.utils.io_utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download or load price dataset.")
    parser.add_argument("--input", type=str,
                        default="../data/sample/prices_sample.csv")
    parser.add_argument("--output", type=str, default="../data/raw/prices/prices.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    ensure_dir(output_path.parent)

    df = load_csv(input_path)
    df.to_csv(output_path, index=False)

    print(f"[✓] Price dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
