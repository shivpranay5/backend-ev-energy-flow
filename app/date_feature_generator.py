from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GeneratedFeatureRow:
    date: str
    match_type: str
    values: dict[str, float]


def _smooth_circular_frame(frame: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    ordered = frame.sort_index()
    pad = window // 2
    extended = pd.concat([ordered.tail(pad), ordered, ordered.head(pad)])
    smoothed = extended.rolling(window=window, min_periods=1, center=True).mean()
    return smoothed.iloc[pad : pad + len(ordered)].set_index(ordered.index)


class SeasonalFeatureGenerator:
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        feature_columns: list[str],
        date_col: str = "date",
        monthly_mode: bool = False,
    ) -> None:
        self.frame = frame.copy()
        self.feature_columns = list(feature_columns)
        self.date_col = date_col
        self.monthly_mode = monthly_mode

        self.frame[date_col] = pd.to_datetime(self.frame[date_col]).dt.normalize()
        self.frame["day_of_year"] = self.frame[date_col].dt.dayofyear.astype(int)
        self.frame["day_of_week"] = self.frame[date_col].dt.dayofweek.astype(int)
        self.frame["month"] = self.frame[date_col].dt.month.astype(int)
        self.frame["is_weekend"] = (self.frame["day_of_week"] >= 5).astype(int)

        numeric = self.frame[self.feature_columns].astype(float)
        self.global_mean = numeric.mean()
        self.weekday_delta = (
            self.frame.groupby("day_of_week")[self.feature_columns].mean().subtract(self.global_mean, axis=1)
        )
        self.monthly_profile = self.frame.groupby("month")[self.feature_columns].mean()
        self.doy_profile = _smooth_circular_frame(self.frame.groupby("day_of_year")[self.feature_columns].mean())

    def generate(self, requested_date: str) -> GeneratedFeatureRow:
        target = pd.Timestamp(requested_date).normalize()
        doy = int(target.dayofyear)
        dow = int(target.dayofweek)
        month = int(target.month)
        weekend = 1 if dow >= 5 else 0

        if doy not in self.doy_profile.index:
            match_type = "seasonal_interpolated"
            base = self.monthly_profile.loc[month].copy()
        else:
            match_type = "generated"
            base = self.doy_profile.loc[doy].copy()

        if month in self.monthly_profile.index:
            month_delta = self.monthly_profile.loc[month] - self.global_mean
        else:
            month_delta = 0.0 * self.global_mean

        if dow in self.weekday_delta.index:
            weekday_delta = self.weekday_delta.loc[dow]
        else:
            weekday_delta = 0.0 * self.global_mean

        generated = base + 0.25 * month_delta + 0.35 * weekday_delta
        values = {col: float(generated.get(col, self.global_mean.get(col, 0.0))) for col in self.feature_columns}

        if "day_of_week" in values:
            values["day_of_week"] = float(dow)
        if "month" in values:
            values["month"] = float(month)
        if "day_of_year" in values:
            values["day_of_year"] = float(doy)
        if "is_weekend" in values:
            values["is_weekend"] = float(weekend)

        if "high_price_label" in values and "price_rank" in values:
            values["high_price_label"] = float(1 if values["price_rank"] >= 0.72 else 0)

        return GeneratedFeatureRow(date=target.date().isoformat(), match_type=match_type, values=values)
