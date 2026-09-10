"""Tests for real-data preprocessing loaders and dispatch script."""

from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path
from uuid import uuid4

import pytest

from risk_reasoning.data import medqa
from scripts import prepare_real_data


@pytest.fixture
def scratch_dir() -> Path:
    root = Path(__file__).resolve().parents[1] / ".test_runs" / f"real_data_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_medqa_preprocess_downloads_snapshot_and_normalizes(scratch_dir: Path, monkeypatch) -> None:
    monkeypatch.setenv("RISK_REASONING_DATA_DIR", str(scratch_dir))
    snapshot_calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs):
        snapshot_calls.append(kwargs)
        base = Path(kwargs["local_dir"]) / "med_qa_en_source"
        base.mkdir(parents=True, exist_ok=True)
        (base / "test-00000-of-00001.parquet").write_text("stub", encoding="utf-8")
        (base / "test-00001-of-00002.parquet").write_text("stub", encoding="utf-8")
        return str(base)

    monkeypatch.setattr(medqa, "snapshot_download", fake_snapshot_download)

    def fake_load_dataset(path, *, data_files, split, cache_dir):
        assert path == "parquet"
        assert split == "test"
        assert cache_dir.endswith("datasets")
        split_paths = [Path(path) for path in data_files["test"]]
        assert [path.name for path in split_paths] == [
            "test-00000-of-00001.parquet",
            "test-00001-of-00002.parquet",
        ]
        assert all(path.exists() for path in split_paths)
        return [
            {
                "question": "Most likely diagnosis?",
                "answer_idx": "B",
                "answer": "Pneumonia",
                "options": [
                    {"key": "A", "value": "Asthma"},
                    {"key": "B", "value": "Pneumonia"},
                ],
                "meta_info": "step1",
            },
            {
                "question": "Malformed",
                "answer_idx": "Z",
                "options": [{"key": "A", "value": "Choice"}],
            },
        ]

    _install_fake_datasets(monkeypatch, fake_load_dataset)

    cfg = {
        "name": "medqa",
        "hf_path": "bigbio/med_qa",
        "hf_config": "med_qa_en_source",
        "hf_revision": "known-good",
        "split": "test",
        "max_items": 10,
    }

    out_path = medqa.preprocess(cfg)
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["id"] == "medqa-test-0"
    assert rows[0]["answer"]["choice_id"] == "B"
    assert rows[0]["answer"]["explanation"] == "Pneumonia"
    assert snapshot_calls[0]["allow_patterns"] == ["med_qa_en_source/*", "med_qa_en_source/**"]
    assert snapshot_calls[0]["revision"] == "known-good"

    loaded = list(medqa.load({**cfg, "max_items": 1}))
    assert len(loaded) == 1
    assert loaded[0].choices is not None
    assert loaded[0].choices[1].text == "Pneumonia"


def test_medqa_split_files_support_split_subdirectories(scratch_dir: Path) -> None:
    base = scratch_dir / "med_qa_en_source"
    split_dir = base / "test"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "0000.parquet").write_text("stub", encoding="utf-8")

    split_files = medqa._split_files(base, "test")

    assert [path.name for path in split_files] == ["0000.parquet"]


def test_medqa_preprocess_validates_required_config_keys() -> None:
    with pytest.raises(ValueError, match=r"\[medqa\] Missing required config key: 'hf_config'"):
        medqa.preprocess(
            {
                "name": "medqa",
                "hf_path": "bigbio/med_qa",
                "split": "test",
            }
        )


def test_prepare_real_data_main_dispatches_selected_datasets(monkeypatch, capsys) -> None:
    calls: list[str] = []

    def fake_medqa_preprocess(cfg):
        calls.append(cfg["name"])
        return Path("data/processed/medqa.jsonl")

    monkeypatch.setattr(medqa, "preprocess", fake_medqa_preprocess)
    monkeypatch.setattr(
        prepare_real_data,
        "_load_cfg",
        lambda path: {
            "name": path.stem,
            "loader": "risk_reasoning.data.medqa:load",
        },
    )

    rc = prepare_real_data.main(["--datasets", "medqa"])
    captured = capsys.readouterr()

    assert rc == 0
    assert calls == ["medqa"]
    assert "[prepare_real_data] medqa" in captured.out


def test_load_cfg_falls_back_without_omegaconf(monkeypatch, scratch_dir: Path) -> None:
    config_path = scratch_dir / "medqa.yaml"
    config_path.write_text(
        "\n".join(
            [
                'name: medqa',
                'loader: risk_reasoning.data.medqa:load',
                'hf_path: "bigbio/med_qa"',
                'hf_config: "med_qa_en_source"',
                "split: test",
                "max_items: 400",
                "",
                "metrics:",
                "  primary: accuracy",
                "  also: [calibration, reasoning_proxies]",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setitem(sys.modules, "omegaconf", None)
    cfg = prepare_real_data._load_cfg(config_path)

    assert cfg["name"] == "medqa"
    assert cfg["hf_config"] == "med_qa_en_source"
    assert cfg["max_items"] == 400
    assert cfg["metrics"]["also"] == ["calibration", "reasoning_proxies"]


def _install_fake_datasets(monkeypatch, payload):
    if callable(payload):
        fake_load_dataset = payload
    else:
        def fake_load_dataset(*args, **kwargs):
            return payload

    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_module)
