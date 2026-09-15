"""The newest dataset and events file are chosen by the date in the name.

Both previous attempts at "newest" were wrong in a way that looked right:
sorting by name puts events_20200804_... after events_20180914_... although
it is the older scrape, and research_9000rows_... after research_11158rows_...
"""
import os

from tools import data_paths


def _touch(d, *names):
    os.makedirs(d, exist_ok=True)
    for n in names:
        open(os.path.join(d, n), "w").close()


def test_events_file_picks_latest_end_date_not_name_order(tmp_path, monkeypatch):
    monkeypatch.setattr(data_paths, "REPO_ROOT", str(tmp_path))
    _touch(tmp_path / "clusters_history", "events_20180813_20260813.parquet",
           "events_20200804_20260804.parquet", "events_20180914_20260914.parquet",
           "manifest.json")
    assert data_paths.latest_events_file().endswith("events_20180914_20260914.parquet")


def test_events_tie_on_end_date_prefers_widest_window(tmp_path, monkeypatch):
    monkeypatch.setattr(data_paths, "REPO_ROOT", str(tmp_path))
    _touch(tmp_path / "clusters_history", "events_20200914_20260914.parquet",
           "events_20180914_20260914.parquet")
    assert data_paths.latest_events_file().endswith("events_20180914_20260914.parquet")


def test_dataset_picks_latest_date_and_ignores_tagged_variants(tmp_path, monkeypatch):
    monkeypatch.setattr(data_paths, "REPO_ROOT", str(tmp_path))
    _touch(tmp_path / "research_data", "research_9000rows_20260914.parquet",
           "research_11158rows_20260913.parquet",
           "research_groupE_99999rows_20261231.parquet",
           "research_10861rows_20260813.parquet")
    assert data_paths.latest_research_dataset().endswith("research_9000rows_20260914.parquet")


def test_missing_data_falls_back_quietly_unless_required(tmp_path, monkeypatch):
    monkeypatch.setattr(data_paths, "REPO_ROOT", str(tmp_path))
    assert data_paths.latest_research_dataset().endswith("research_10861rows_20260813.parquet")
    try:
        data_paths.latest_events_file(required=True)
    except SystemExit:
        pass
    else:
        raise AssertionError("required=True must fail when nothing matches")
