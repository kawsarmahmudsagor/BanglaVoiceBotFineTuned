from __future__ import annotations

import os
import re

import pandas as pd
from test_select import make_wd

from dataprep.__main__ import main
from dataprep.listen import listen
from dataprep.profiles import Profile, Range
from dataprep.workdir import StepFile

BANDS = [0, 2.5, 3.0, 5]


def _setup(tmp_path, n=30):
    wd = make_wd(tmp_path / "work", n=n)
    clips = pd.read_parquet(wd.clips_path)
    clips["src_path"] = [str(tmp_path / "audio" / f"{i}.wav") for i in clips["id"]]
    clips["text_raw"] = "<b>&"
    StepFile.write(wd.clips_path, clips, step="import", version=1, params={})
    # first half in the 0-2.5 band, second half in 3-5
    ovrl = [2.0 + 0.01 * i if i < n // 2 else 3.5 + 0.01 * i for i in range(n)]
    StepFile.write(
        wd.step_path("dnsmos"),
        pd.DataFrame({
            "id": clips["id"], "dnsmos_p808": 3.0, "dnsmos_sig": 3.0, "dnsmos_bak": 3.0, "dnsmos_ovrl": ovrl, "error": None,
        }),
        step="dnsmos", version=1, params={},
    )
    return wd


def _rows(html_text, band):
    return re.findall(rf'<tr data-band="{band}"', html_text)


def test_listen_per_band_rows(tmp_path):
    wd = _setup(tmp_path)
    prof = Profile(name="t", task="tts", filters=[Range(column="dnsmos_ovrl", min=4.0)])
    out = listen(wd, prof, per_band=4, bands=BANDS)
    assert out == wd.root / "listen" / "t.html"
    text = out.read_text(encoding="utf-8")
    # the filter on the listened column is ignored so the whole range is covered
    assert len(_rows(text, "0-2.5")) == 4
    assert len(_rows(text, "2.5-3")) == 0
    assert len(_rows(text, "3-5")) == 4


def test_listen_deterministic_and_relative_links(tmp_path):
    wd = _setup(tmp_path)
    prof = Profile(name="t", task="tts")
    a = listen(wd, prof, per_band=3, bands=BANDS).read_text(encoding="utf-8")
    b = listen(wd, prof, per_band=3, bands=BANDS).read_text(encoding="utf-8")
    assert a == b
    srcs = re.findall(r'<audio[^>]* src="([^"]+)"', a)
    assert len(srcs) == 6
    html_dir = wd.root / "listen"
    for s in srcs:
        assert not os.path.isabs(s) and s.startswith("..")
        assert (html_dir / s).resolve() == (tmp_path / "audio" / os.path.basename(s)).resolve()


def test_listen_escapes_text_and_shows_scores(tmp_path):
    wd = _setup(tmp_path, n=6)
    text = listen(wd, Profile(name="t", task="tts"), per_band=2, bands=BANDS).read_text(encoding="utf-8")
    assert "&lt;b&gt;&amp;" in text and "<b>&" not in text
    assert "dnsmos_ovrl" in text


def test_listen_cli(tmp_path):
    wd = _setup(tmp_path, n=6)
    prof = tmp_path / "p.yaml"
    prof.write_text("name: p\ntask: tts\n", encoding="utf-8")
    assert main(["listen", "--work", str(wd.root), "--profile", str(prof), "--per-band", "1"]) == 0
    assert (wd.root / "listen" / "p.html").is_file()


def test_listen_shipped_tts_loose_without_speakers_or_labeller(tmp_path, capsys):
    wd = _setup(tmp_path, n=12)
    assert main(["listen", "--work", str(wd.root), "--profile", "tts-loose", "--per-band", "2"]) == 0
    out = capsys.readouterr().out
    assert "CER filter skipped" in out and "speaker floor skipped" in out
    assert (wd.root / "listen" / "tts-loose.html").is_file()


def test_listen_rejects_non_score_column(tmp_path, capsys):
    wd = _setup(tmp_path, n=6)
    assert main(["listen", "--work", str(wd.root), "--profile", "tts-loose", "--column", "dur"]) == 1
    assert "cannot listen by 'dur'" in capsys.readouterr().out
