"""
scheduler.py
============
PPLE Investment Club · Algo Team
-------------------------------------------------
Runs the full trading pipeline once per trading day at a configurable time.
Designed to run continuously on a server (e.g. a small VPS or Raspberry Pi).

Default schedule: 09:35 ET (5 minutes after NYSE open)
 — early enough to act on overnight moves
 — late enough for opening volatility to settle

Usage:
  python scheduler.py                  # run daily at 09:35 ET
  python scheduler.py --time 15:55     # run at 15:55 ET (near close)
  python scheduler.py --run-now        # execute once immediately, then schedule
  python scheduler.py --no-execute     # dry run (no orders submitted)
-------------------------------------------------
"""

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta

import pytz

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)

EASTERN = pytz.timezone("US/Eastern")

# NYSE trading days are Mon–Fri; we skip weekends
# (A full implementation would also skip US market holidays)
TRADING_DAYS = {0, 1, 2, 3, 4}  # Mon=0 … Fri=4


def is_trading_day(dt: datetime) -> bool:
    return dt.weekday() in TRADING_DAYS


def next_run_time(run_hour: int, run_minute: int) -> datetime:
    """
    Return the next datetime (in ET) when the pipeline should run.
    If today's run time has already passed, schedule for the next trading day.
    """
    now_et = datetime.now(EASTERN)
    candidate = now_et.replace(hour=run_hour, minute=run_minute, second=0, microsecond=0)

    # If time already passed today, move to tomorrow
    if now_et >= candidate:
        candidate += timedelta(days=1)

    # Skip weekends
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)

    return candidate


def run_pipeline(extra_args: list[str] = None) -> None:
    """Invoke main.py as a subprocess so it runs in a clean context."""
    cmd = [sys.executable, "main.py"] + (extra_args or [])
    logger.info(f"[SCHEDULER] Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error(f"[SCHEDULER] main.py exited with code {result.returncode}")
    else:
        logger.info("[SCHEDULER] Pipeline completed successfully.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPLE Algo daily scheduler")
    p.add_argument("--time",       default="09:35", metavar="HH:MM",
                   help="Daily run time in US/Eastern (default: 09:35)")
    p.add_argument("--run-now",    action="store_true",
                   help="Execute pipeline once immediately before entering loop.")
    p.add_argument("--no-execute", action="store_true",
                   help="Pass --no-execute to main.py (dry run).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    hour, minute = map(int, args.time.split(":"))
    extra = ["--no-execute"] if args.no_execute else []

    logger.info("=" * 55)
    logger.info("  PPLE ALGO · SCHEDULER STARTED")
    logger.info(f"  Daily run: {args.time} US/Eastern (Mon–Fri)")
    logger.info("=" * 55)

    if args.run_now:
        logger.info("[SCHEDULER] --run-now flag: executing immediately...")
        run_pipeline(extra)

    while True:
        next_run = next_run_time(hour, minute)
        wait_sec = (next_run - datetime.now(EASTERN)).total_seconds()

        logger.info(
            f"[SCHEDULER] Next run: {next_run.strftime('%Y-%m-%d %H:%M %Z')} "
            f"({wait_sec/3600:.1f}h from now)"
        )
        time.sleep(max(0, wait_sec))

        logger.info("[SCHEDULER] Wake — running pipeline...")
        run_pipeline(extra)

        # Small buffer to avoid re-triggering in the same minute
        time.sleep(90)


if __name__ == "__main__":
    main()
