"""tests/unit/test_evaluator.py"""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model.evaluator import (
    winkler_score, coverage_hit, evaluate_predictions,
    format_metrics_display, predictions_to_dataframe,
    compute_calibration_curve, decompose_winkler,
)


class TestWinklerScore:
    def test_hit_inside_interval(self):
        assert winkler_score(100, 95, 105) == 10

    def test_hit_at_bounds(self):
        assert winkler_score(95, 95, 105) == 10
        assert winkler_score(105, 95, 105) == 10

    def test_miss_below(self):
        score = winkler_score(90, 95, 105)
        assert score > 10
        assert score == 10 + (2 / 0.05) * (95 - 90)

    def test_miss_above(self):
        score = winkler_score(110, 95, 105)
        assert score > 10
        assert score == 10 + (2 / 0.05) * (110 - 105)

    def test_custom_alpha(self):
        score = winkler_score(90, 95, 105, alpha=0.10)
        assert score == 10 + (2 / 0.10) * (95 - 90)


class TestCoverageHit:
    def test_inside(self):
        assert coverage_hit(100, 95, 105) == 1

    def test_at_bounds(self):
        assert coverage_hit(95, 95, 105) == 1
        assert coverage_hit(105, 95, 105) == 1

    def test_below(self):
        assert coverage_hit(90, 95, 105) == 0

    def test_above(self):
        assert coverage_hit(110, 95, 105) == 0


class TestEvaluatePredictions:
    def test_empty(self):
        r = evaluate_predictions([])
        assert r["coverage_95"] == 0.0
        assert r["total_predictions"] == 0

    def test_all_hits(self):
        preds = [
            {"actual_close": 100, "predicted_low_95": 95, "predicted_high_95": 105},
            {"actual_close": 100, "predicted_low_95": 95, "predicted_high_95": 105},
        ]
        r = evaluate_predictions(preds)
        assert r["coverage_95"] == 1.0
        assert r["total_hits"] == 2
        assert r["total_misses"] == 0

    def test_all_misses(self):
        preds = [
            {"actual_close": 80, "predicted_low_95": 95, "predicted_high_95": 105},
            {"actual_close": 120, "predicted_low_95": 95, "predicted_high_95": 105},
        ]
        r = evaluate_predictions(preds)
        assert r["coverage_95"] == 0.0
        assert r["total_misses"] == 2


class TestFormatMetricsDisplay:
    def test_format(self):
        m = {
            "coverage_95": 0.9528,
            "avg_width_95": 1225.83,
            "mean_winkler_95": 1683.47,
            "total_predictions": 720,
            "total_hits": 686,
            "total_misses": 34,
        }
        f = format_metrics_display(m)
        assert f["coverage"] == "95.3%"
        assert f["avg_width"] == "$1,225.83"
        assert f["winkler"] == "1,683.47"
        assert f["predictions"] == "720"
        assert f["hits"] == "686"
        assert f["misses"] == "34"


class TestPredictionsToDataframe:
    def test_basic(self):
        preds = [
            {
                "bar_timestamp": "2026-05-01 00:00:00",
                "actual_close": 100,
                "predicted_low_95": 95,
                "predicted_high_95": 105,
            }
        ]
        df = predictions_to_dataframe(preds)
        assert len(df) == 1
        assert "width" in df.columns
        assert "hit" in df.columns
        assert "winkler" in df.columns

    def test_empty(self):
        df = predictions_to_dataframe([])
        assert len(df) == 0


class TestComputeCalibrationCurve:
    def test_50_hits(self):
        preds = [
            {"actual_close": 100, "predicted_low_95": 90, "predicted_high_95": 110}
            for _ in range(50)
        ]
        cc = compute_calibration_curve(preds)
        assert len(cc["expected"]) == 7
        assert 0.9 <= cc["actual"][4] <= 1.0

    def test_empty(self):
        cc = compute_calibration_curve([])
        assert cc["expected"] == [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


class TestDecomposeWinkler:
    def test_all_hits(self):
        preds = [
            {"actual_close": 100, "predicted_low_95": 95, "predicted_high_95": 105}
            for _ in range(10)
        ]
        wd = decompose_winkler(preds)
        assert wd["n_hits"] == 10
        assert wd["n_misses"] == 0
        assert wd["penalty_ratio"] == 0.0

    def test_mixed(self):
        preds = [
            {"actual_close": 100, "predicted_low_95": 95, "predicted_high_95": 105}
            if i < 7 else
            {"actual_close": 200, "predicted_low_95": 95, "predicted_high_95": 105}
            for i in range(10)
        ]
        wd = decompose_winkler(preds)
        assert wd["n_hits"] == 7
        assert wd["n_misses"] == 3
        assert wd["penalty_ratio"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])