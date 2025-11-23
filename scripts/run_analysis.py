
from pathlib import Path
import pandas as pd
from fns_project.data.loader import load_financial_data
from fns_project.features.indicators import compute_additional_indicators
from fns_project.viz.plots import plot_price_with_all_indicators

if __name__ == "__main__":
    path = Path("data/raw/price_sample.csv")
    df = load_financial_data(path, date_col="Date", tz="Etc/GMT+4")

    df_ind = compute_additional_indicators(df)

    plot_price_with_all_indicators(df_ind)

    print(df_ind.tail(5))
