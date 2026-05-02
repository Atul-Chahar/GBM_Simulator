"""
storage.py — Prediction Persistence (Part C)
==============================================
Stores prediction history so that returning visitors see
a growing timeline of predictions with actuals filled in.

Strategy: JSON file-based storage with Google Sheets as optional
persistent backend for Streamlit Community Cloud (ephemeral filesystem).

On Streamlit Cloud, we use gspread + Google Sheets as the database.
Locally, we fall back to a local JSON file.
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional

import pandas as pd

# ── Local JSON storage path ──────────────────────────────────────
LOCAL_STORAGE_FILE = "prediction_history.json"


class PredictionStore:
    """
    Manages persistent prediction history.

    Supports two backends:
    1. Local JSON file (for development / non-cloud)
    2. Google Sheets via gspread (for Streamlit Community Cloud)
    """

    def __init__(self, use_gsheets: bool = False, sheet_url: str = None):
        """
        Parameters
        ----------
        use_gsheets : bool
            If True, use Google Sheets for persistence.
        sheet_url : str
            Google Sheets URL (required if use_gsheets=True).
        """
        self.use_gsheets = use_gsheets
        self.sheet_url = sheet_url
        self._sheet = None

        if use_gsheets and sheet_url:
            self._init_gsheets()

    def _init_gsheets(self):
        """Initialize Google Sheets connection."""
        try:
            import gspread
            from google.oauth2.service_account import Credentials
            import streamlit as st

            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ]

            # Load credentials from Streamlit secrets
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
            client = gspread.authorize(creds)
            self._sheet = client.open_by_url(self.sheet_url).sheet1

            # Ensure headers exist
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
        """
        Save a new prediction to the store.

        Parameters
        ----------
        prediction : dict
            Must contain: timestamp, current_price, predicted_low_95,
            predicted_high_95. actual_close and hit are filled later.
        """
        ts_raw = prediction.get("timestamp", datetime.now(timezone.utc).isoformat())
        try:
            ts_iso = pd.to_datetime(ts_raw).tz_convert('UTC').isoformat()
        except Exception:
            ts_iso = str(ts_raw)
            
        record = {
            "timestamp": ts_iso,
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
            self._save_to_local(record)

    def get_history(self) -> List[Dict]:
        """
        Retrieve all saved predictions.

        Returns
        -------
        list[dict]
            Prediction history, newest first.
        """
        if self.use_gsheets and self._sheet:
            return self._load_from_gsheets()
        else:
            return self._load_from_local()

    def verify_predictions(self, current_prices: Dict[str, float]) -> int:
        """
        Check unverified predictions against actual prices.

        Parameters
        ----------
        current_prices : dict
            Mapping of timestamp → actual close price.

        Returns
        -------
        int
            Number of predictions verified.
        """
        history = self.get_history()
        verified_count = 0

        for record in history:
            if record.get("verified"):
                continue

            ts = record.get("timestamp", "")
            try:
                # Normalize to standard ISO string to handle Google Sheets formatting quirks (e.g. replacing T with space)
                ts_iso = pd.to_datetime(ts).tz_convert('UTC').isoformat()
            except Exception:
                ts_iso = str(ts)
                
            if ts_iso in current_prices:
                actual = current_prices[ts_iso]
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

        # Save back
        if verified_count > 0:
            if self.use_gsheets and self._sheet:
                self._overwrite_gsheets(history)
            else:
                self._overwrite_local(history)

        return verified_count

    def get_history_dataframe(self) -> pd.DataFrame:
        """
        Get prediction history as a pandas DataFrame.
        """
        history = self.get_history()
        if not history:
            return pd.DataFrame(columns=[
                "timestamp", "current_price", "predicted_low_95",
                "predicted_high_95", "actual_close", "hit", "winkler",
                "verified"
            ])

        df = pd.DataFrame(history)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.sort_values("timestamp", ascending=False, inplace=True)

        return df

    def should_save_new_prediction(self) -> bool:
        """
        Check if enough time has passed since last prediction
        to avoid duplicate saves on page refreshes.

        Returns True if > 30 minutes since last prediction.
        """
        history = self.get_history()
        if not history:
            return True

        # Get latest prediction timestamp
        latest = max(history, key=lambda x: x.get("timestamp", ""))
        try:
            last_time = datetime.fromisoformat(
                latest["timestamp"].replace("Z", "+00:00")
            )
            now = datetime.now(timezone.utc)
            elapsed = (now - last_time).total_seconds()
            return elapsed > 1800  # 30 minutes
        except (ValueError, KeyError):
            return True

    # ── Local JSON backend ───────────────────────────────────────

    def _save_to_local(self, record: Dict) -> None:
        history = self._load_from_local()
        history.append(record)
        self._overwrite_local(history)

    def _load_from_local(self) -> List[Dict]:
        if not os.path.exists(LOCAL_STORAGE_FILE):
            return []
        try:
            with open(LOCAL_STORAGE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _overwrite_local(self, history: List[Dict]) -> None:
        with open(LOCAL_STORAGE_FILE, "w") as f:
            json.dump(history, f, indent=2, default=str)

    # ── Google Sheets backend ────────────────────────────────────

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
            self._save_to_local(record)

    def _load_from_gsheets(self) -> List[Dict]:
        try:
            records = self._sheet.get_all_records()
            return records
        except Exception as e:
            print(f"⚠️ Google Sheets read failed: {e}")
            return self._load_from_local()

    def _overwrite_gsheets(self, history: List[Dict]) -> None:
        try:
            # Clear and rewrite (simple approach for small datasets)
            self._sheet.clear()
            headers = [
                "timestamp", "current_price", "predicted_low_95",
                "predicted_high_95", "actual_close", "hit", "winkler",
                "verified"
            ]
            self._sheet.append_row(headers)
            for record in history:
                self._sheet.append_row([
                    str(record.get(h, "")) for h in headers
                ])
        except Exception as e:
            print(f"⚠️ Google Sheets overwrite failed: {e}")


# ─── Quick self-test ─────────────────────────────────────────────
if __name__ == "__main__":
    store = PredictionStore(use_gsheets=False)

    # Save a test prediction
    store.save_prediction({
        "current_price": 97432.50,
        "predicted_low_95": 97100.00,
        "predicted_high_95": 97800.00,
    })

    print("History:")
    for p in store.get_history():
        print(f"  {p}")

    print(f"\nShould save new: {store.should_save_new_prediction()}")
