"""Download the two weekly exports (prices + offers) into DATA_DIR/inbox."""

import gzip
import logging
from datetime import datetime
from pathlib import Path

import httpx

from flats_pipeline.config import settings

logger = logging.getLogger(__name__)


def _download(url: str, out: Path) -> Path | None:
    tmp = out.with_suffix(".part")
    try:
        with httpx.stream(
            "GET",
            url,
            headers=settings.DOWNLOAD_HEADERS,
            follow_redirects=True,
            timeout=httpx.Timeout(60, read=600),
        ) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        # Make sure we got a complete gzip, not an HTML login page or a cut-off file.
        with gzip.open(tmp, "rb") as g:
            while g.read(1 << 24):
                pass
    except Exception as e:  # noqa: BLE001
        logger.error("Download failed for %s: %s: %s", out.name, type(e).__name__, e)
        tmp.unlink(missing_ok=True)
        return None
    tmp.replace(out)
    logger.info("Downloaded %s (%.1f MB)", out.name, out.stat().st_size / 2**20)
    return out


def download_weekly() -> list[Path]:
    targets = {"offer_pub_list": settings.PRICES_URL, "offer": settings.OFFERS_URL}
    if not any(targets.values()):
        logger.info("No download URLs configured, using files already in inbox folders")
        return []
    settings.download_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    done = []
    for name, url in targets.items():
        if url:
            out = settings.download_dir / f"for-analytics.msk_habit.{name}_{stamp}.csv.gz"
            if (p := _download(url, out)) is not None:
                done.append(p)
    return done
