"""Discriminate actual-file verification from matching untrusted ledgers."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("compare_builds", Path(__file__).parents[1] / "tools/compare_builds.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_tree(root):
    (root / "results").mkdir()
    (root / "results/value.json").write_bytes(b"verified bytes")
    ledger = {"results/value.json": hashlib.sha256(b"verified bytes").hexdigest()}
    (root / "artifacts.sha256.json").write_text(json.dumps(ledger))
    return ledger


def test_rehash_valid_complete_tree(tmp_path):
    expected = fixture_tree(tmp_path)
    assert module.verified_inventory(tmp_path) == expected


@pytest.mark.parametrize("damage", ["missing", "corrupt", "extra", "empty"])
def test_reject_untrustworthy_ledger(tmp_path, damage):
    fixture_tree(tmp_path)
    if damage == "missing":
        (tmp_path / "results/value.json").unlink()
    elif damage == "corrupt":
        (tmp_path / "results/value.json").write_bytes(b"changed bytes")
    elif damage == "extra":
        (tmp_path / "results/unlisted.json").write_bytes(b"extra")
    else:
        (tmp_path / "artifacts.sha256.json").write_text("{}")
    with pytest.raises(ValueError):
        module.verified_inventory(tmp_path)
