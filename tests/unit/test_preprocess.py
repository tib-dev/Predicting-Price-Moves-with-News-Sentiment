import pandas as pd
from src.fns_project.data import preprocess


def test_clean_and_metrics():
    df = pd.DataFrame(
        {"headline": ["Apple hits $AAPL high! Read more: https://x.co/test", None]})
    cleaned = preprocess.preprocess_headlines(
        df, text_col="headline", drop_empty=True)
    assert not cleaned.empty
    cleaned = preprocess.add_headline_metrics(cleaned)
    assert "headline_len_chars" in cleaned.columns
    assert "headline_word_count" in cleaned.columns
