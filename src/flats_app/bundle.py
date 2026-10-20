"""Loads the precomputed bundle (APP_DATA_DIR) written by `python -m flats_pipeline export-app`."""

import json
from dataclasses import dataclass
from pathlib import Path

from catboost import CatBoostRegressor

from flats_app.local import LocalPrice

SUPPORTED_FORMAT = 2


@dataclass
class Bundle:
    manifest: dict
    model: CatBoostRegressor
    meta: dict
    periods: list[str]
    areas: dict[tuple[str, str], dict]
    local: LocalPrice

    @property
    def months(self) -> list[str]:
        return self.meta["months"]

    def area(self, level: str, name: str | None) -> dict | None:
        return self.areas.get((level, name)) if name else None


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_bundle(data_dir: Path, threads: int = 2) -> Bundle:
    if not (data_dir / "manifest.json").exists():
        raise RuntimeError(
            f"No app data in {data_dir}. Export it with: python -m flats_pipeline export-app"
        )
    manifest = _json(data_dir / "manifest.json")
    if manifest.get("format_version") != SUPPORTED_FORMAT:
        raise RuntimeError(f"Unsupported bundle format {manifest.get('format_version')} (need {SUPPORTED_FORMAT})")
    missing = [f for f in manifest["files"] if not (data_dir / f).exists()]
    if missing:
        raise RuntimeError(f"Bundle in {data_dir} is incomplete, missing: {missing}")

    meta = _json(data_dir / "model" / "model_meta.json")
    model = CatBoostRegressor(thread_count=threads)
    model.load_model(str(data_dir / "model" / "model.cbm"))
    indices = _json(data_dir / "indices" / "areas.json")
    if indices["periods"] != [m.replace("_", "-") for m in meta["months"]]:
        raise RuntimeError("Bundle indices and model cover different months; re-export it")
    return Bundle(
        manifest=manifest,
        model=model,
        meta=meta,
        periods=indices["periods"],
        areas={(a["level"], a["area"]): a for a in indices["areas"]},
        local=LocalPrice.from_npz(
            data_dir / "local" / "local_price_reference.npz", k=meta["local_k_neighbors"], workers=threads
        ),
    )
