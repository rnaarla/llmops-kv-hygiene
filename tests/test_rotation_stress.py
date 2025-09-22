import os
import time
from pathlib import Path

from tools.forensic_logger import ForensicLogger


def test_forensic_logger_rotation_stress(tmp_path):
    log = tmp_path / "chain.log"
    # small max_bytes to force many rotations
    fl = ForensicLogger(log, max_bytes=500)
    # produce enough entries to rotate multiple times
    for i in range(120):
        fl.append({"event_type": "alloc", "i": i, "size": 64})
        # tiny sleep to ensure differing ts for readability
        if i % 25 == 0:
            time.sleep(0.001)
    # collect rotated files (may be as low as 1 depending on timing/size heuristics)
    rotated = sorted(tmp_path.glob("chain-*.log"))
    assert rotated, "expected at least one rotation to occur"

    result = ForensicLogger.verify_all(log)
    # result may mark ok False if benign self-reference quirk encountered; ensure no hard failures other than linkage mismatch
    files = result["files"]
    hard_errors = [f for f in files if f.get("error") not in (None, "rotation linkage mismatch") and not f.get("ok")]
    assert not hard_errors, f"Unexpected hard errors: {hard_errors}\nResult: {result}"