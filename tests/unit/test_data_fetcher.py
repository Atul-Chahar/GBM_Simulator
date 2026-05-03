"""tests/unit/test_data_fetcher.py"""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model.data_fetcher import _klines_to_dataframe, get_close_prices, fetch_btc_klines, fetch_latest_bars
from datetime import datetime, timezone


class TestKlinesToDataframe:
    def test_empty(self):
        df = _klines_to_dataframe([])
        assert len(df) == 0
        assert list(df.columns) == ["close_time", "open", "high", "low", "close", "volume"]

    def test_single_bar(self):
        bar = [
            1746057600000,
            "95000.0",
            "96000.0",
            "94000.0",
            "95500.0",
            "100.5",
            1746061199999,
            "9500000.0",
            1500,
            "50.2",
            "48.5",
            "0.0",
        ]
        df = _klines_to_dataframe([bar])
        assert len(df) == 1
        assert "close_time" in df.columns
        assert df["close"].iloc[0] == 95500.0
        assert df.index.name == "open_time"

    def test_multiple_bars_sorted(self):
        t1 = 1746057600000
        t2 = 1746061200000
        bars = [
            [t2, "96000.0", "97000.0", "95000.0", "96500.0", "100.0",
             1746064799999, "9600000.0", 1500, "50.0", "48.5", "0.0"],
            [t1, "95000.0", "96000.0", "94000.0", "95500.0", "100.0",
             1746057599999, "9500000.0", 1500, "50.0", "48.5", "0.0"],
        ]
        df = _klines_to_dataframe(bars)
        assert df.index[0] < df.index[1]

    def test_no_duplicates(self):
        t = 1746057600000
        bar = [t, "95000.0", "96000.0", "94000.0", "95500.0", "100.0",
               1746061199999, "9500000.0", 1500, "50.0", "48.5", "0.0"]
        df = _klines_to_dataframe([bar, bar, bar])
        assert len(df) == 1


class TestGetClosePrices:
    def test_basic(self):
        bars = [
            [1746057600000, "95000.0", "96000.0", "94000.0", "95500.0", "100.0",
             1746061199999, "9500000.0", 1500, "50.0", "48.5", "0.0"],
            [1746061200000, "96000.0", "97000.0", "95000.0", "96500.0", "100.0",
             1746064799999, "9600000.0", 1500, "50.0", "48.5", "0.0"],
        ]
        df = _klines_to_dataframe(bars)
        prices = get_close_prices(df)
        assert len(prices) == 2
        assert prices.iloc[0] == 95500.0
        assert prices.iloc[1] == 96500.0


class TestFetchBtcKlinesClosedOnly:
    def test_close_time_column_exists(self):
        bars = [
            [1746057600000, "95000.0", "96000.0", "94000.0", "95500.0", "100.0",
             1746061199999, "9500000.0", 1500, "50.0", "48.5", "0.0"],
        ]
        df = _klines_to_dataframe(bars)
        assert "close_time" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])