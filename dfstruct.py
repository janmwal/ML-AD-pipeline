"""DFStruct container providing reproducible access to precomputed dataframes."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Tuple

import pandas as pd


class DFStruct:
    """Lightweight keyed storage for preprocessed DataFrame variants."""

    def __init__(self, drop_for_training: Optional[Iterable[str]] = None):
        self._store: Dict[Tuple[Tuple[str, Any], ...], pd.DataFrame] = {}
        self._train_exclude = set(
            drop_for_training
            or ["OASISID", "seg_days", "cdr_days", "delta_days", "CDRTOT"]
        )

    # ---------- internals ----------
    def _norm_key(self, kwargs: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
        return tuple(sorted(kwargs.items()))

    def _drop_cols(
        self, df: pd.DataFrame, extra_drop: Optional[Iterable[str]] = None
    ) -> pd.DataFrame:
        cols = self._train_exclude.union(extra_drop or [])
        return df.drop(columns=list(cols), errors="ignore")

    # ---------- core API ----------
    def __call__(self, **kwargs) -> pd.DataFrame:
        return self._store[self._norm_key(kwargs)]

    def add(self, df: pd.DataFrame, **kwargs) -> None:
        self._store[self._norm_key(kwargs)] = df

    def filter(
        self, **kwargs
    ) -> Iterator[Tuple[Tuple[Tuple[str, Any], ...], pd.DataFrame]]:
        for key, df in self._store.items():
            if all(item in dict(key).items() for item in kwargs.items()):
                yield key, df

    def apply(
        self, func: Callable[[pd.DataFrame], pd.DataFrame], **kwargs
    ) -> None:
        for key, df in self.filter(**kwargs):
            self._store[key] = func(df)

    # ---------- training helpers ----------
    def get_for_training(
        self,
        *,
        extra_drop: Optional[Iterable[str]] = None,
        copy_df: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        df = self(**kwargs)
        df = deepcopy(df) if copy_df else df
        return self._drop_cols(df, extra_drop=extra_drop)

    def prepare_for_training_inplace(
        self, *, extra_drop: Optional[Iterable[str]] = None, **kwargs
    ) -> None:
        def _strip(d: pd.DataFrame) -> pd.DataFrame:
            return self._drop_cols(d, extra_drop=extra_drop)

        self.apply(_strip, **kwargs)

    def set_training_drop_columns(self, cols: Iterable[str]) -> None:
        self._train_exclude = set(cols)

    def add_training_drop_columns(self, cols: Iterable[str]) -> None:
        self._train_exclude.update(cols)

    # ---------- key inspection ----------
    def keys_raw(self) -> list[tuple[tuple[str, Any], ...]]:
        return list(self._store.keys())

    def keys_dict(self) -> list[dict[str, Any]]:
        return [dict(k) for k in self._store.keys()]

    def flag_space(self) -> dict[str, set]:
        out: Dict[str, set] = {}
        for k in self._store.keys():
            for name, val in k:
                out.setdefault(name, set()).add(val)
        return out
