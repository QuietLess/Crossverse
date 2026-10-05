"""Data contracts for the two core tables.

interactions: user_id, item_id, domain, rating, timestamp
items:        item_id, domain, title, text, themes, genres, year, popularity
"""

from __future__ import annotations

import pandas as pd

from crossverse import DOMAINS

INTERACTION_COLUMNS = ["user_id", "item_id", "domain", "rating", "timestamp"]
ITEM_COLUMNS = ["item_id", "domain", "title", "text", "themes", "genres", "year"]


class DataContractError(ValueError):
    pass


def validate_interactions(df: pd.DataFrame, items: pd.DataFrame | None = None) -> None:
    missing = set(INTERACTION_COLUMNS) - set(df.columns)
    if missing:
        raise DataContractError(f"interactions missing columns: {sorted(missing)}")
    if df[INTERACTION_COLUMNS].isna().any().any():
        raise DataContractError("interactions contain nulls in required columns")
    if not df["rating"].between(1, 5).all():
        raise DataContractError("ratings must lie in [1, 5]")
    if not df["domain"].isin(DOMAINS).all():
        raise DataContractError(f"domain must be one of {DOMAINS}")
    if (df["timestamp"] <= 0).any():
        raise DataContractError("timestamps must be positive epoch values")
    if df.duplicated(["user_id", "item_id"]).any():
        raise DataContractError("duplicate (user_id, item_id) pairs; aggregate before saving")
    if items is not None:
        unknown = ~df["item_id"].isin(items["item_id"])
        if unknown.any():
            raise DataContractError(f"{int(unknown.sum())} interactions reference unknown items")


def validate_items(df: pd.DataFrame) -> None:
    missing = set(ITEM_COLUMNS) - set(df.columns)
    if missing:
        raise DataContractError(f"items missing columns: {sorted(missing)}")
    if df["item_id"].duplicated().any():
        raise DataContractError("duplicate item_id in catalog")
    if not df["domain"].isin(DOMAINS).all():
        raise DataContractError(f"domain must be one of {DOMAINS}")
    if df["title"].isna().any() or (df["title"].str.len() == 0).any():
        raise DataContractError("every item needs a title")
