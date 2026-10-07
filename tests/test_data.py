"""Data layer: packaged reference points and the locator of the Letter data tables."""
import json
import os

import pytest

from dscorr import data


def test_reference_points_content():
    r = data.reference_points()
    counts = {k: len(r[k]) for k in ("double", "triple", "single", "scalene", "mass_range",
                                      "squeezed")}
    assert counts == {"double": 12, "triple": 8, "single": 12, "scalene": 20, "mass_range": 14,
                      "squeezed": 10}
    assert r["tested_range"] == {"mu_min": 0.3, "mu_max": 4.0, "momentum_ratio_min": 0.002}
    for row in r["scalene"] + r["double"] + r["triple"]:
        k = row["k"] if "k" in row else (row["k1"], row["k2"], row["k3"])
        assert abs(row["F_direct_SK"] - 8 * row["S_direct_SK"] / (k[0] * k[1] * k[2])) \
            <= 1e-15 * abs(row["F_direct_SK"])
    sing = [x for x in r["mass_range"] if x["diagram"] == "single"]
    assert all(len(x["channels"]) == 3 for x in sing)


def _make_letter(tmp, name="triangle_conformal_33config.csv"):
    d = tmp / "examples" / "Letter" / "data"
    d.mkdir(parents=True)
    (d / name).write_text("tag\n")
    (d / "x.json").write_text(json.dumps({"points": []}))
    return d


def test_letter_dir_env(tmp_path, monkeypatch):
    d = _make_letter(tmp_path)
    monkeypatch.setenv("DSCORR_DATA", str(d))
    assert data.letter_dir() == str(d.resolve())


def test_letter_dir_cwd_and_parents(tmp_path, monkeypatch):
    monkeypatch.delenv("DSCORR_DATA", raising=False)
    d = _make_letter(tmp_path)
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    for wd in (tmp_path, tmp_path / "a", sub):
        monkeypatch.chdir(wd)
        found = data.letter_dir(required=False)
        # a source checkout would take precedence over the working directory
        assert found is not None


def test_letter_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("DSCORR_DATA", str(tmp_path / "nowhere"))
    monkeypatch.chdir(tmp_path)
    if data.letter_dir(required=False) is None:
        with pytest.raises(FileNotFoundError, match="Letter data not found; set DSCORR_DATA"):
            data.letter_dir()


def test_load_json_and_csv(tmp_path, monkeypatch):
    d = _make_letter(tmp_path)
    (d / "t.csv").write_text("# comment\n# more\na,b,c\n1.5,,x\n2,3,y\n")
    monkeypatch.setenv("DSCORR_DATA", str(d))
    assert data.load("x.json") == {"points": []}
    rows = data.load("t.csv")
    assert rows == [{"a": 1.5, "b": None, "c": "x"}, {"a": 2.0, "b": 3.0, "c": "y"}]
    with pytest.raises(FileNotFoundError):
        data.load("missing.json")
