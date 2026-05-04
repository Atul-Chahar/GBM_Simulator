"""
storage.py — Prediction Persistence (Part C)
==============================================
Stores prediction history so that returning visitors see
a growing timeline of predictions with actuals filled in.

Uses SQLite for concurrent-safe, atomic persistence.
Google Sheets remains as optional cloud backup.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import List, Dict

import pandas as pd

LOCAL_STORAGE_FILE = "prediction_history.json"
DB_FILE = "predictions.db"


def _is_numeric_string(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _migrate_json_to_sqlite(json_file: str, db_file: str) -> None:
    """One-time migration of JSON history to SQLite."""
    if not os.path.exists(json_file):
        return
    try:
        with open(json_file, "r") as f:
            history = json.load(f)
        os.remove(json_file)
    except (json.JSONDecodeError, IOError):
        return

    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER UNIQUE NOT NULL,
                current_price REAL,
                predicted_low_95 REAL,
                predicted_high_95 REAL,
                actual_close REAL,
                hit INTEGER,
                winkler REAL,
                verified INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
    for record in history:
            ts_str = record.get("timestamp", "")
            cp = record.get("current_price", 0)
            ac = record.get("actual_close")
            if cp == 0 and (ac == 0 or ac is None):
                continue
            if isinstance(ac, str) and not _is_numeric_string(ac):
                continue
            try:
                ts = pd.to_datetime(ts_str, utc=True)
                ts_ms = int(ts.timestamp() * 1000)
                if ts_ms > datetime.now(timezone.utc).timestamp() * 1000 + 3600_000:
                    continue
            except Exception:
                continue
            c.execute("""
                INSERT OR IGNORE INTO predictions
                (timestamp, current_price, predicted_low_95, predicted_high_95,
                 actual_close, hit, winkler, verified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ts_ms,
                record.get("current_price", 0),
                record.get("predicted_low_95", 0),
                record.get("predicted_high_95", 0),
                record.get("actual_close"),
                record.get("hit"),
                record.get("winkler"),
                1 if record.get("verified") else 0,
            ))
    conn.commit()
    conn.close()


class PredictionStore:
    """
    Manages persistent prediction history using SQLite.

    Supports two backends:
    1. SQLite database (default — concurrent-safe, atomic)
    2. Google Sheets via gspread (optional cloud backup)
    """

    def __init__(self, use_gsheets: bool = False, sheet_url: str = None):
        self.use_gsheets = use_gsheets
        self.sheet_url = sheet_url
        self._sheet = None
        self._db_file = DB_FILE
        self._init_sqlite()

        if use_gsheets and sheet_url:
            self._init_gsheets()

        self._validate_and_purge()

    def _init_sqlite(self) -> None:
        _migrate_json_to_sqlite(LOCAL_STORAGE_FILE, self._db_file)
        conn = sqlite3.connect(self._db_file)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER UNIQUE NOT NULL,
                current_price REAL,
                predicted_low_95 REAL,
                predicted_high_95 REAL,
                actual_close REAL,
                hit INTEGER,
                winkler REAL,
                verified INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_file)

    def _init_gsheets(self):
        try:
            import gspread
            from google.oauth2.service_account import Credentials
            import streamlit as st

            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ]

            if not hasattr(st, 'secrets') or "gcp_service_account" not in st.secrets:
                raise KeyError("gcp_service_account not found in st.secrets")

            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
            client = gspread.authorize(creds)
            self._sheet = client.open_by_url(self.sheet_url).sheet1

            if not self._sheet.row_values(1):
                self._sheet.append_row([
                    "timestamp", "current_price", "predicted_low_95",
                    "predicted_high_95", "actual_close", "hit", "winkler",
                    "verified"
                ])

        except Exception as e:
            print(f"⚠️ Google Sheets init failed: {e}. Falling back to local.")
            self.use_gsheets = False

    def save_prediction(self, prediction: Dict) -> None:
        ts_raw = prediction.get("timestamp", datetime.now(timezone.utc).timestamp() * 1000)
        try:
            ts_ms = int(pd.to_datetime(ts_raw).timestamp() * 1000) if isinstance(ts_raw, str) else int(ts_raw)
        except Exception:
            ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        record = {
            "timestamp": ts_ms,
            "current_price": prediction.get("current_price", 0),
            "predicted_low_95": prediction.get("predicted_low_95", 0),
            "predicted_high_95": prediction.get("predicted_high_95", 0),
            "actual_close": prediction.get("actual_close", None),
            "hit": prediction.get("hit", None),
            "winkler": prediction.get("winkler", None),
            "verified": prediction.get("verified", False),
        }

        if self.use_gsheets and self._sheet:
            self._save_to_gsheets(record)
        else:
            self._save_to_sqlite(record)

    def get_history(self) -> List[Dict]:
        if self.use_gsheets and self._sheet:
            return self._load_from_gsheets()
        return self._load_from_sqlite()

    def _load_from_sqlite(self) -> List[Dict]:
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        rows = c.execute(
            "SELECT * FROM predictions ORDER BY timestamp DESC"
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def _save_to_sqlite(self, record: Dict) -> None:
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO predictions
            (timestamp, current_price, predicted_low_95, predicted_high_95,
             actual_close, hit, winkler, verified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["timestamp"],
            record["current_price"],
            record["predicted_low_95"],
            record["predicted_high_95"],
            record["actual_close"],
            record["hit"],
            record["winkler"],
            1 if record["verified"] else 0,
        ))
        conn.commit()
        conn.close()

    def verify_predictions(self, current_prices: Dict[int, float]) -> int:
        history = self.get_history()
        verified_count = 0

        for record in history:
            if record.get("verified"):
                continue

            ts = record.get("timestamp", 0)
            if not isinstance(ts, int):
                try:
                    ts = int(pd.to_datetime(ts).timestamp() * 1000)
                except Exception:
                    continue

            if ts in current_prices:
                actual = current_prices[ts]
                low = record["predicted_low_95"]
                high = record["predicted_high_95"]

                record["actual_close"] = actual
                record["hit"] = int(low <= actual <= high)

                width = high - low
                if actual < low:
                    record["winkler"] = width + (2 / 0.05) * (low - actual)
                elif actual > high:
                    record["winkler"] = width + (2 / 0.05) * (actual - high)
                else:
                    record["winkler"] = width

                record["verified"] = True
                verified_count += 1

                self._update_sqlite(record)

        return verified_count

    def _update_sqlite(self, record: Dict) -> None:
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("""
            UPDATE predictions SET
                actual_close = ?, hit = ?, winkler = ?, verified = ?
            WHERE timestamp = ?
        """, (
            record["actual_close"],
            record["hit"],
            record["winkler"],
            1 if record["verified"] else 0,
            record["timestamp"],
        ))
        conn.commit()
        conn.close()

    def get_history_dataframe(self) -> pd.DataFrame:
        history = self.get_history()
        if not history:
            return pd.DataFrame(columns=[
                "timestamp", "current_price", "predicted_low_95",
                "predicted_high_95", "actual_close", "hit", "winkler",
                "verified"
            ])

        df = pd.DataFrame(history)
        if "timestamp" in df.columns:
            ts_col = df["timestamp"]
            if ts_col.dtype == 'int64' or isinstance(ts_col.iloc[0], int):
                df["timestamp"] = pd.to_datetime(ts_col, unit='ms', utc=True)
            else:
                df["timestamp"] = pd.to_datetime(ts_col, errors="coerce", utc=True)
            df.dropna(subset=["timestamp"], inplace=True)
            df.sort_values("timestamp", ascending=False, inplace=True)

        return df

    def should_save_new_prediction(self) -> bool:
        history = self.get_history()
        if not history:
            return True

        latest = max(history, key=lambda x: x.get("timestamp", 0))
        try:
            ts_val = latest["timestamp"]
            if isinstance(ts_val, int):
                last_time = pd.to_datetime(ts_val, unit='ms', utc=True)
            else:
                last_time = pd.to_datetime(ts_val, utc=True)
            now = datetime.now(timezone.utc)
            elapsed = (now - last_time).total_seconds()
            return elapsed > 1800
        except (ValueError, KeyError):
            return True

    def _validate_and_purge(self) -> None:
        conn = self._get_conn()
        c = conn.cursor()
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        c.execute("SELECT COUNT(*) FROM predictions")
        total = c.fetchone()[0]
        if total == 0:
            conn.close()
            return

        c.execute("DELETE FROM predictions WHERE current_price = 0 AND actual_close IS NULL")
        c.execute("DELETE FROM predictions WHERE actual_close = 0 AND hit IS NULL")
        c.execute("DELETE FROM predictions WHERE timestamp > ?", (now_ms,))

        deleted = c.execute("SELECT changes()").fetchone()[0]
        conn.commit()
        conn.close()

    # ── Google Sheets backend ───────────────────────────────────

    def _save_to_gsheets(self, record: Dict) -> None:
        try:
            self._sheet.append_row([
                str(record.get("timestamp", "")),
                record.get("current_price", 0),
                record.get("predicted_low_95", 0),
                record.get("predicted_high_95", 0),
                record.get("actual_close", ""),
                record.get("hit", ""),
                record.get("winkler", ""),
                str(record.get("verified", False)),
            ])
        except Exception as e:
            print(f"⚠️ Google Sheets write failed: {e}. Saving locally.")
            self._save_to_sqlite(record)

    def _load_from_gsheets(self) -> List[Dict]:
        try:
            records = self._sheet.get_all_records()
            return records
        except Exception as e:
            print(f"⚠️ Google Sheets read failed: {e}")
            return self._load_from_sqlite()


if __name__ == "__main__":
    store = PredictionStore(use_gsheets=False)

    store.save_prediction({
        "current_price": 97432.50,
        "predicted_low_95": 97100.00,
        "predicted_high_95": 97800.00,
    })

    print("History:")
    for p in store.get_history():
        print(f"  {p}")

    print(f"\nShould save new: {store.should_save_new_prediction()}")