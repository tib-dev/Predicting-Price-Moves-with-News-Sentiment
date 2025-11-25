"""Generate HTML/PDF reports for financial data and sentiment."""

import logging
from pathlib import Path
import pandas as pd
import panel as pn

from fns_project.viz.plots import create_price_sentiment_dashboard

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

pn.extension("plotly", template="bootstrap")


def generate_html_report(
    report_path: str | Path,
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    price_summary_cols: list[str] = None,
    sentiment_summary_cols: list[str] = None,
    title: str = "Financial & Sentiment Report"
):
    """
    Generate a simple HTML report with:
    - Price + SMA dashboard
    - Sentiment + rolling dashboard
    - Summary tables for price & sentiment
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Create dashboards
    dashboard = create_price_sentiment_dashboard(
        price_df, sentiment_df,
        sma_cols=[c for c in (price_summary_cols or [])
                  if c in price_df.columns],
        rolling_cols=[c for c in (
            sentiment_summary_cols or []) if c in sentiment_df.columns],
        title=title
    )

    # Price summary table
    price_summary = price_df.describe().reset_index(
    ) if price_summary_cols is None else price_df[price_summary_cols].describe().reset_index()

    # Sentiment summary table
    sentiment_summary = sentiment_df.describe().reset_index(
    ) if sentiment_summary_cols is None else sentiment_df[sentiment_summary_cols].describe().reset_index()

    # Build Panel template
    report = pn.Column(
        pn.pane.Markdown(f"# {title}"),
        dashboard,
        pn.pane.Markdown("## Price Summary"),
        pn.widgets.DataFrame(price_summary, width=800, height=200),
        pn.pane.Markdown("## Sentiment Summary"),
        pn.widgets.DataFrame(sentiment_summary, width=800, height=200)
    )

    # Save as standalone HTML
    report.save(str(report_path), embed=True, resources="inline")
    logger.info("Report saved to %s", report_path)
    return report_path


