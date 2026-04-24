"""
src/logging_setup.py
====================
Centralised logging configuration for the project.

Provides:
  • Console handler  (INFO+ colourised)
  • File handler     (DEBUG+ plain-text, written to logs/)
  • setup_logging()  – call once at entry point

Colour codes (ANSI) are stripped when output is not a TTY so log files
remain clean.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


# ── ANSI colour helpers ───────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREY   = "\033[90m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_BRED   = "\033[1;31m"

_LEVEL_COLORS = {
    "DEBUG":    _GREY,
    "INFO":     _GREEN,
    "WARNING":  _YELLOW,
    "ERROR":    _RED,
    "CRITICAL": _BRED,
}


class _ColorFormatter(logging.Formatter):
    """Colourised formatter for TTY output."""

    FMT = "{color}{levelname:<8}{reset}  {cyan}{name}{reset}  {msg}"

    def format(self, record: logging.LogRecord) -> str:
        use_color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        if use_color:
            color = _LEVEL_COLORS.get(record.levelname, "")
            prefix = self.FMT.format(
                color=color,
                levelname=record.levelname,
                reset=_RESET,
                cyan=_CYAN,
                name=record.name,
                msg="",
            )
            return prefix + super().format(record)
        return super().format(record)


class _PlainFormatter(logging.Formatter):
    """Plain formatter for file output."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
def setup_logging(
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    run_name: str | None = None,
) -> logging.Logger:
    """
    Configure the root logger with a console handler and a rotating file
    handler.

    Parameters
    ----------
    log_dir       : directory where .log files are written
    console_level : minimum level shown on stdout  (default INFO)
    file_level    : minimum level written to file  (default DEBUG)
    run_name      : optional run identifier; used in the log filename

    Returns
    -------
    Root logger instance
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix   = f"_{run_name}" if run_name else ""
    log_file = Path(log_dir) / f"run{suffix}_{ts}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # handlers filter individually

    # Remove any previously added handlers (e.g. from Jupyter re-runs)
    root.handlers.clear()

    # ── Console ───────────────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(
        _ColorFormatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(ch)

    # ── File ──────────────────────────────────────────────────────────────
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(
        _PlainFormatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(fh)

    root.info("Logging initialised  |  file → %s", log_file)
    return root
