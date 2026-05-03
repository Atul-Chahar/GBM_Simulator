"""tests/unit/test_storage.py"""
import pytest
import sys, os, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from persistence.storage import PredictionStore, _is_numeric_string, _migrate_json_to_sqlite
import sqlite3


class TestIsNumericString:
    def test_numeric_strings(self):
        assert _is_numeric_string("123.45") is True
        assert _is_numeric_string("0") is True
        assert _is_numeric_string("-123.45") is True

    def test_non_numeric_strings(self):
        assert _is_numeric_string("TEST_CONNECTION_FINAL") is False
        assert _is_numeric_string("abc") is False
        assert _is_numeric_string("") is False


class TestPredictionStoreSQLite:
    @pytest.fixture
    def tmp_db(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_save_and_retrieve(self, tmp_db):
        import persistence.storage as s
        orig = s.DB_FILE
        s.DB_FILE = tmp_db
        try:
            store = PredictionStore(use_gsheets=False)
            store.save_prediction({
                "timestamp": "2026-05-01T12:00:00+00:00",
                "current_price": 95000.0,
                "predicted_low_95": 94000.0,
                "predicted_high_95": 96000.0,
            })
            history = store.get_history()
            assert len(history) == 1
            assert history[0]["current_price"] == 95000.0
        finally:
            s.DB_FILE = orig

    def test_should_save_first(self, tmp_db):
        import persistence.storage as s
        orig = s.DB_FILE
        s.DB_FILE = tmp_db
        try:
            store = PredictionStore(use_gsheets=False)
            assert store.should_save_new_prediction() is True
        finally:
            s.DB_FILE = orig

    def test_should_not_save_too_soon(self, tmp_db):
        import persistence.storage as s
        orig = s.DB_FILE
        s.DB_FILE = tmp_db
        try:
            store = PredictionStore(use_gsheets=False)
            store.save_prediction({
                "timestamp": "2026-05-01T12:00:00+00:00",
                "current_price": 95000.0,
                "predicted_low_95": 94000.0,
                "predicted_high_95": 96000.0,
            })
            assert store.should_save_new_prediction() is False
        finally:
            s.DB_FILE = orig

    def test_get_history_dataframe(self, tmp_db):
        import persistence.storage as s
        orig = s.DB_FILE
        s.DB_FILE = tmp_db
        try:
            store = PredictionStore(use_gsheets=False)
            store.save_prediction({
                "timestamp": "2026-05-01T12:00:00+00:00",
                "current_price": 95000.0,
                "predicted_low_95": 94000.0,
                "predicted_high_95": 96000.0,
            })
            df = store.get_history_dataframe()
            assert len(df) == 1
            assert "timestamp" in df.columns
            assert "current_price" in df.columns
        finally:
            s.DB_FILE = orig

    def test_purge_invalid(self, tmp_db):
        import persistence.storage as s
        orig = s.DB_FILE
        s.DB_FILE = tmp_db
        try:
            conn = sqlite3.connect(tmp_db)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT, current_price REAL, predicted_low_95 REAL,
                    predicted_high_95 REAL, actual_close REAL, hit INTEGER,
                    winkler REAL, verified INTEGER DEFAULT 0
                )
            """)
            c.execute("INSERT INTO predictions (timestamp, current_price, predicted_low_95, predicted_high_95) VALUES (?, ?, ?, ?)",
                ("2026-05-01T12:00:00+00:00", 0, 0, 0))
            c.execute("INSERT INTO predictions (timestamp, current_price, predicted_low_95, predicted_high_95) VALUES (?, ?, ?, ?)",
                ("2026-05-01T13:00:00+00:00", 95000.0, 94000.0, 96000.0))
            conn.commit()
            conn.close()

            store = PredictionStore(use_gsheets=False)
            history = store.get_history()
            assert len(history) == 1
            assert history[0]["current_price"] == 95000.0
        finally:
            s.DB_FILE = orig


if __name__ == "__main__":
    pytest.main([__file__, "-v"])