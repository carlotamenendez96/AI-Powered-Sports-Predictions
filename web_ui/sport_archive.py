"""Soft-delete for the basketball blueprints (Euroleague, NBA).

Mirrors football's ``/football/delete_file``: the file moves to
``<out_dir>/history/`` rather than being removed. ``compute_sport_summary``
reads ``history/`` too, so an archived slip still counts in the lane totals;
it only leaves the visible lists.
"""

from __future__ import annotations

import datetime
import fnmatch
import json
import os
import shutil
from typing import Tuple


def archive_file(out_dir: str, filename: str, allowed: Tuple[str, ...]) -> Tuple[bool, str]:
    """Move ``out_dir/filename`` to ``out_dir/history/``. Returns (ok, message).

    Only names matching one of ``allowed`` (glob patterns) are accepted.
    Bet slips must be CLOSED: settlement only looks in ``out_dir``, so an
    archived OPEN slip would never settle — the same rule football enforces.
    """
    if os.path.basename(filename) != filename or '..' in filename:
        return False, 'Invalid filename.'
    if not any(fnmatch.fnmatch(filename, p) for p in allowed):
        return False, f'{filename} cannot be archived here.'
    src = os.path.join(out_dir, filename)
    if not os.path.exists(src):
        return False, f'{filename} not found.'

    if filename.startswith('bets_'):
        try:
            with open(src) as f:
                status = (json.load(f) or {}).get('status')
        except (OSError, ValueError):
            status = None
        if status != 'CLOSED':
            return False, f'{filename} is not settled yet — archive is available after settlement.'

    hist = os.path.join(out_dir, 'history')
    os.makedirs(hist, exist_ok=True)
    target = os.path.join(hist, filename)
    if os.path.exists(target):
        base, ext = os.path.splitext(filename)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        target = os.path.join(hist, f'{base}.{ts}{ext}')
    try:
        shutil.move(src, target)
    except OSError as e:
        return False, f'Error archiving {filename}: {e}'
    return True, f'Archived {filename} (moved to history/).'
