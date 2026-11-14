"""Tests for retest reconciliation: matching rules and browser workflow."""

from __future__ import annotations

import io
import threading
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from app import create_app
from app import services
from app.db import connect
from app.reconcile import (
    ParameterError,
    PublishedReading,
    RetestReading,
    parse_deviation,
    parse_time_micros,
    parse_time_window_minutes,
    reconcile,
)
from app.validation import validate_file


@pytest.fixture()
def app(tmp_path: Path):
    application = create_app(
        {
            "TESTING": True,
            "DATA_DIR": str(tmp_path),
            "DATABASE": str(tmp_path / "test.db"),
            "FILES_DIR": str(tmp_path / "files"),
            "SECRET_KEY": "test-secret",
            "CSRF_ENABLED": False,
        }
    )
    yield application


@pytest.fixture()
def client(app):
    return app.test_client()


def reading(site, metric, at, value, row_no=1, kind="retest"):
    cls = RetestReading if kind == "retest" else PublishedReading
    return cls(
        row_no=row_no,
        site=site,
        metric=metric,
        observed_at=at,
        value_text=str(value),
        value=Decimal(str(value)),
        time_micros=parse_time_micros(at),
    )


def run(retest, published, window_minutes=30, deviation="0.5"):
    window, micros = parse_time_window_minutes(str(window_minutes))
    return reconcile(
        retest,
        published,
        window_micros=micros,
        max_abs_deviation=parse_deviation(deviation),
        max_time_diff_minutes=window,
    )


def make_csv(rows, header="站点,指标,采样时刻,参照值"):
    return ("\n".join([header, *[",".join(row) for row in rows]]) + "\n").encode(
        "utf-8"
    )


def upload_and_publish(client, app, rows, dataset_date="2026-11-14"):
    content = make_csv(
        [(dataset_date, *row) for row in rows],
        header="dataset_date,site,metric,observed_at,value",
    )
    response = client.post(
        "/admin/upload",
        data={"dataset_date": dataset_date, "file": (io.BytesIO(content), "obs.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 302, response.get_data(as_text=True)
    conn = connect(app.config["DATABASE"])
    try:
        version = conn.execute(
            "SELECT * FROM versions WHERE dataset_date = ? ORDER BY version_no DESC LIMIT 1",
            (dataset_date,),
        ).fetchone()
    finally:
        conn.close()
    client.post(f"/admin/versions/{version['id']}/publish")
    return version


def post_retest(
    client,
    rows,
    *,
    dataset_date="2026-11-14",
    window="30",
    deviation="0.5",
    content=None,
    name="retest.csv",
):
    return client.post(
        "/admin/reconcile",
        data={
            "dataset_date": dataset_date,
            "max_time_diff": window,
            "max_abs_deviation": deviation,
            "file": (io.BytesIO(content if content is not None else make_csv(rows)), name),
        },
        content_type="multipart/form-data",
    )


# ---------------------------------------------------------------------------
# Matching rules
# ---------------------------------------------------------------------------


def test_exact_matches_pair_one_to_one():
    retest = [
        reading("A", "氨氮", "2026-11-14 08:00:00", "1.00", 1),
        reading("A", "氨氮", "2026-11-14 09:00:00", "2.00", 2),
    ]
    published = [
        reading("A", "氨氮", "2026-11-14 08:00:00", "1.10", 1, "published"),
        reading("A", "氨氮", "2026-11-14 09:00:00", "2.10", 2, "published"),
    ]
    report = run(retest, published)
    assert report.pair_count == 2
    assert report.total_time_diff_micros == 0
    assert not report.retest_unmatched
    assert not report.published_unmatched


def test_metric_and_site_must_match():
    report = run(
        [reading("A", "氨氮", "2026-11-14 08:00:00", "1")],
        [
            reading("A", "总磷", "2026-11-14 08:00:00", "1", kind="published"),
            reading("B", "氨氮", "2026-11-14 08:00:00", "1", kind="published"),
        ],
    )
    assert report.pair_count == 0
    assert len(report.retest_unmatched) == 1
    assert len(report.published_unmatched) == 2


def test_time_window_is_enforced():
    report = run(
        [reading("A", "氨氮", "2026-11-14 08:00:00", "1")],
        [reading("A", "氨氮", "2026-11-14 08:31:00", "1", kind="published")],
        window_minutes=30,
    )
    assert report.pair_count == 0

    report = run(
        [reading("A", "氨氮", "2026-11-14 08:00:00", "1")],
        [reading("A", "氨氮", "2026-11-14 08:30:00", "1", kind="published")],
        window_minutes=30,
    )
    assert report.pair_count == 1


def test_maximises_pair_count_before_time_cost():
    """Greedy nearest-neighbour would strand one reading; max cardinality wins."""
    retest = [
        reading("A", "氨氮", "2026-11-14 08:00:00", "1", 1),
        reading("A", "氨氮", "2026-11-14 08:10:00", "1", 2),
    ]
    published = [
        reading("A", "氨氮", "2026-11-14 08:06:00", "1", 1, "published"),
        reading("A", "氨氮", "2026-11-14 08:14:00", "1", 2, "published"),
    ]
    report = run(retest, published, window_minutes=20)
    assert report.pair_count == 2
    # Optimal pairing: 08:00->08:06 (6 min) + 08:10->08:14 (4 min) = 10 min.
    assert report.total_time_diff_micros == 10 * 60 * 1_000_000


def test_minimises_total_time_difference_among_maximum_matchings():
    retest = [
        reading("A", "氨氮", "2026-11-14 08:00:00", "1", 1),
        reading("A", "氨氮", "2026-11-14 08:05:00", "1", 2),
    ]
    published = [
        reading("A", "氨氮", "2026-11-14 08:04:00", "1", 1, "published"),
        reading("A", "氨氮", "2026-11-14 08:06:00", "1", 2, "published"),
    ]
    report = run(retest, published, window_minutes=30)
    assert report.pair_count == 2
    # Optimal: 08:00->08:04 (4 min) + 08:05->08:06 (1 min) = 5 min.
    # The crossed pairing would cost 6 + 4 = 10 minutes.
    assert report.total_time_diff_micros == 5 * 60 * 1_000_000


def test_no_reading_is_used_twice():
    retest = [
        reading("A", "氨氮", "2026-11-14 08:00:00", "1", 1),
        reading("A", "氨氮", "2026-11-14 08:01:00", "1", 2),
    ]
    published = [reading("A", "氨氮", "2026-11-14 08:00:30", "1", 1, "published")]
    report = run(retest, published)
    assert report.pair_count == 1
    assert len(report.retest_unmatched) == 1
    assert report.retest_unmatched[0][1] == "时间差阈值内的已发布读数均已被其他复测读数配对"


def test_duplicate_readings_pair_by_minimum_total_cost():
    retest = [
        reading("A", "氨氮", "2026-11-14 08:00:00", "1.0", 1),
        reading("A", "氨氮", "2026-11-14 08:00:00", "1.02", 2),
    ]
    published = [
        reading("A", "氨氮", "2026-11-14 08:03:00", "1.01", 1, "published"),
        reading("A", "氨氮", "2026-11-14 08:09:00", "1.00", 2, "published"),
    ]
    report = run(retest, published, window_minutes=20)
    assert report.pair_count == 2
    assert report.total_time_diff_micros == (3 + 9) * 60 * 1_000_000


def test_deviation_limit_reports_overflow_without_blocking_pairs():
    report = run(
        [reading("A", "氨氮", "2026-11-14 08:00:00", "1.00")],
        [reading("A", "氨氮", "2026-11-14 08:00:00", "1.60", kind="published")],
        deviation="0.5",
    )
    assert report.pair_count == 1
    assert len(report.over_limit) == 1
    assert report.over_limit[0].deviation_text == "0.6"
    assert report.over_limit[0].excess_text == "0.1"

    exactly_at_limit = run(
        [reading("A", "氨氮", "2026-11-14 08:00:00", "1.00")],
        [reading("A", "氨氮", "2026-11-14 08:00:00", "1.50", kind="published")],
        deviation="0.5",
    )
    assert not exactly_at_limit.over_limit


def test_decimal_arithmetic_avoids_float_rounding():
    report = run(
        [reading("A", "氨氮", "2026-11-14 08:00:00", "0.30")],
        [reading("A", "氨氮", "2026-11-14 08:00:00", "0.10", kind="published")],
        deviation="0.2",
    )
    assert not report.over_limit
    assert report.pairs[0].deviation_text == "0.2"


def test_groups_are_independent_and_readings_stay_unique_across_sites():
    retest = [
        reading("A", "氨氮", "2026-11-14 08:00:00", "1", 1),
        reading("A", "氨氮", "2026-11-14 08:00:00", "1", 2),
        reading("B", "氨氮", "2026-11-14 08:00:00", "1", 3),
    ]
    published = [
        reading("A", "氨氮", "2026-11-14 08:00:00", "1", 1, "published"),
        reading("B", "氨氮", "2026-11-14 08:00:00", "1", 2, "published"),
    ]
    report = run(retest, published)
    assert report.pair_count == 2
    assert len(report.retest_unmatched) == 1


def test_parameters_reject_invalid_values():
    for raw in ("", "abc", "-1", "NaN", "inf"):
        with pytest.raises(ParameterError):
            parse_time_window_minutes(raw)
    for raw in ("", "x", "-0.1"):
        with pytest.raises(ParameterError):
            parse_deviation(raw)
    assert parse_time_window_minutes(" 15.5 ") == (Decimal("15.5"), 930_000_000)
    with pytest.raises(ParameterError):
        parse_time_window_minutes("1441")


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------


def test_retest_file_allows_repeated_site_metric_time(tmp_path):
    from app.validation import (
        RETEST_FIELD_LABELS,
        RETEST_HEADER_ALIASES,
        validate_file,
    )

    path = tmp_path / "retest.csv"
    path.write_text(
        "站点,指标,采样时刻,参照值\n"
        "东闸断面,氨氮,2026-11-14 08:00,1.24\n"
        "东闸断面,氨氮,2026-11-14 08:00,1.26\n",
        encoding="utf-8",
    )
    result = validate_file(
        path,
        date(2026, 11, 14),
        allow_duplicate_keys=True,
        header_aliases=RETEST_HEADER_ALIASES,
        field_labels=RETEST_FIELD_LABELS,
    )
    assert result.ok, result.issues
    assert len(result.observations) == 2


def test_retest_file_rejects_bad_rows(tmp_path):
    from app.validation import (
        RETEST_FIELD_LABELS,
        RETEST_HEADER_ALIASES,
        validate_file,
    )

    path = tmp_path / "retest.csv"
    path.write_text(
        "站点,指标,采样时刻,参照值\n"
        "东闸断面,氨氮,2026-11-14 08:00,abc\n"
        "东闸断面,氨氮,2026-11-13 08:00,1.2\n",
        encoding="utf-8",
    )
    result = validate_file(
        path,
        date(2026, 11, 14),
        allow_duplicate_keys=True,
        header_aliases=RETEST_HEADER_ALIASES,
        field_labels=RETEST_FIELD_LABELS,
    )
    assert not result.ok
    codes = {issue.code for issue in result.issues}
    assert {"invalid_value", "date_mismatch"} <= codes


# ---------------------------------------------------------------------------
# Browser workflow
# ---------------------------------------------------------------------------


def test_reconcile_form_is_reachable(client):
    response = client.get("/admin/reconcile")
    assert response.status_code == 200
    assert "最大时间差" in response.get_data(as_text=True)


def test_reconcile_end_to_end_lists_all_sections(client, app):
    version = upload_and_publish(
        client,
        app,
        [
            ("东闸断面", "氨氮", "2026-11-14 08:00:00", "1.24"),
            ("东闸断面", "pH", "2026-11-14 08:00:00", "7.31"),
            ("西渡断面", "氨氮", "2026-11-14 14:00:00", "0.86"),
        ],
    )
    rows = [
        ("东闸断面", "氨氮", "2026-11-14 08:05", "1.30"),
        ("东闸断面", "pH", "2026-11-14 09:30", "7.90"),
        ("北港断面", "氨氮", "2026-11-14 08:00", "2.00"),
    ]
    response = post_retest(client, rows, window="30", deviation="0.1")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert f"v{version['version_no']}" in body
    assert "配对结果" in body
    assert "绝对偏差超限项" in body
    assert "复测侧未配对" in body
    assert "已发布侧未配对" in body
    # 08:05 pairs with 08:00 (5 min); 09:30 is outside the window.
    assert "东闸断面" in body and "1.3" in body
    # Over-limit: the paired 氨氮 reading deviates 0.06 > 0.1? no -> within.
    assert "在限" in body
    assert "北港断面" in body
    assert "西渡断面" in body


def test_reconcile_marks_over_limit(client, app):
    upload_and_publish(
        client, app, [("东闸断面", "氨氮", "2026-11-14 08:00:00", "1.24")]
    )
    response = post_retest(
        client,
        [("东闸断面", "氨氮", "2026-11-14 08:00", "2.00")],
        window="30",
        deviation="0.5",
    )
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "超限" in body
    assert "0.76" in body


def test_reconcile_without_published_version_explains_and_returns_no_result(client, app):
    content = make_csv([("东闸断面", "氨氮", "2026-11-20 08:00", "1.2")])
    response = post_retest(
        client, [], content=content, dataset_date="2026-11-20"
    )
    assert response.status_code == 409
    body = response.get_data(as_text=True)
    assert "没有已发布的版本" in body
    assert "配对结果" not in body


def test_candidate_only_date_is_not_usable(client, app):
    content = make_csv(
        [("2026-11-14", "东闸断面", "氨氮", "2026-11-14 08:00:00", "1.24")],
        header="dataset_date,site,metric,observed_at,value",
    )
    client.post(
        "/admin/upload",
        data={"dataset_date": "2026-11-14", "file": (io.BytesIO(content), "o.csv")},
        content_type="multipart/form-data",
    )
    response = post_retest(
        client, [("东闸断面", "氨氮", "2026-11-14 08:00", "1.2")]
    )
    assert response.status_code == 409
    assert "没有已发布的版本" in response.get_data(as_text=True)


def test_reconcile_invalid_file_reports_issues_without_result(client, app):
    upload_and_publish(
        client, app, [("东闸断面", "氨氮", "2026-11-14 08:00:00", "1.24")]
    )
    content = make_csv([("东闸断面", "氨氮", "2026-11-14 08:00", "abc")])
    response = post_retest(client, [], content=content)
    assert response.status_code == 422
    body = response.get_data(as_text=True)
    assert "数值无效" in body
    assert "配对结果" not in body


def test_reconcile_invalid_parameters_do_not_run(client, app):
    upload_and_publish(
        client, app, [("东闸断面", "氨氮", "2026-11-14 08:00:00", "1.24")]
    )
    response = post_retest(
        client, [("东闸断面", "氨氮", "2026-11-14 08:00", "1.2")], window="-5"
    )
    assert response.status_code == 400
    assert "不能为负数" in response.get_data(as_text=True)
    assert "配对结果" not in response.get_data(as_text=True)


def test_reconcile_missing_file_is_explained(client, app):
    upload_and_publish(
        client, app, [("东闸断面", "氨氮", "2026-11-14 08:00:00", "1.24")]
    )
    response = client.post(
        "/admin/reconcile",
        data={"dataset_date": "2026-11-14", "max_time_diff": "30", "max_abs_deviation": "0.5"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "请选择要上传的复测" in response.get_data(as_text=True)


def test_reconcile_requires_csrf_when_enabled(tmp_path):
    application = create_app(
        {
            "DATA_DIR": str(tmp_path),
            "DATABASE": str(tmp_path / "t.db"),
            "FILES_DIR": str(tmp_path / "files"),
            "SECRET_KEY": "s",
        }
    )
    client = application.test_client()
    response = client.post(
        "/admin/reconcile",
        data={"dataset_date": "2026-11-14"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_reconcile_uses_snapshot_not_a_mix_of_versions(client, app):
    """A re-publish between snapshot read and rendering must not mix versions."""
    upload_and_publish(
        client, app, [("东闸断面", "氨氮", "2026-11-14 08:00:00", "1.24")]
    )
    conn = connect(app.config["DATABASE"])
    try:
        snapshot = services.load_published_snapshot(conn, "2026-11-14")
    finally:
        conn.close()
    assert snapshot.version["version_no"] == 1
    assert [row["value_text"] for row in snapshot.rows] == ["1.24"]

    # Publish a second version (adds a pH row) after the snapshot was taken.
    second = upload_and_publish(
        client,
        app,
        [
            ("东闸断面", "氨氮", "2026-11-14 08:00:00", "1.24"),
            ("东闸断面", "pH", "2026-11-14 08:00:00", "7.31"),
        ],
    )
    assert second["version_no"] == 2

    # The already-taken snapshot is still exclusively version 1's detail.
    assert [row["value_text"] for row in snapshot.rows] == ["1.24"]
    assert len(snapshot.rows) == snapshot.version["row_count"] == 1


def test_snapshot_returns_no_rows_when_detail_is_incomplete(client, app):
    upload_and_publish(
        client, app, [("东闸断面", "氨氮", "2026-11-14 08:00:00", "1.24")]
    )
    conn = connect(app.config["DATABASE"])
    try:
        conn.execute("DELETE FROM observations")
        conn.commit()
        snapshot = services.load_published_snapshot(conn, "2026-11-14")
    finally:
        conn.close()
    assert snapshot.version is not None
    assert snapshot.rows == []
    assert "明细不完整" in snapshot.error


def test_concurrent_republish_during_reconcile_yields_one_consistent_version(
    client, app, monkeypatch
):
    """Interleave a publish between the read and the response.

    ``load_published_snapshot`` is wrapped so the publish happens right after
    the snapshot is materialised; the rendered result must still belong to the
    snapshot version only.
    """
    upload_and_publish(
        client, app, [("东闸断面", "氨氮", "2026-11-14 08:00:00", "1.24")]
    )
    real_loader = services.load_published_snapshot

    def loader_then_republish(conn, dataset_date):
        snapshot = real_loader(conn, dataset_date)
        # Re-publish v2 with an extra reading, using an independent connection.
        worker = app.test_client()
        content = make_csv(
            [
                ("2026-11-14", "东闸断面", "氨氮", "2026-11-14 08:00:00", "1.24"),
                ("2026-11-14", "东闸断面", "pH", "2026-11-14 08:00:00", "7.31"),
            ],
            header="dataset_date,site,metric,observed_at,value",
        )
        worker.post(
            "/admin/upload",
            data={
                "dataset_date": "2026-11-14",
                "file": (io.BytesIO(content), "obs.csv"),
            },
            content_type="multipart/form-data",
        )
        conn2 = connect(app.config["DATABASE"])
        try:
            newest = conn2.execute(
                "SELECT * FROM versions WHERE dataset_date = ? ORDER BY version_no DESC LIMIT 1",
                ("2026-11-14",),
            ).fetchone()
        finally:
            conn2.close()
        worker.post(f"/admin/versions/{newest['id']}/publish")
        return snapshot

    monkeypatch.setattr(services, "load_published_snapshot", loader_then_republish)

    response = post_retest(
        client,
        [("东闸断面", "氨氮", "2026-11-14 08:00", "1.24")],
        window="30",
        deviation="0.5",
    )
    assert response.status_code == 200
    body = response.get_data(as_text=True)

    # Result came from v1 (1 detail row), even though v2 is now published.
    assert "v1" in body
    assert "已重新发布" in body
    # v2's pH reading must not leak into the published-side detail.
    assert "7.31" not in body
    assert "已发布第 2 行" in body

    conn = connect(app.config["DATABASE"])
    try:
        published = conn.execute(
            "SELECT version_no FROM versions WHERE dataset_date = ? AND status = 'published'",
            ("2026-11-14",),
        ).fetchall()
    finally:
        conn.close()
    assert [row["version_no"] for row in published] == [2]


def test_snapshot_stays_internally_consistent_under_concurrent_publishes(app):
    """Every snapshot read during repeated publishes is exactly one version.

    The snapshots must always contain exactly the ``row_count`` rows of the
    version they name — never a blend of the old and new version.
    """
    worker = app.test_client()
    upload_and_publish(
        worker, app, [("东闸断面", "氨氮", "2026-11-14 08:00:00", "1.24")]
    )

    stop = threading.Event()
    problems: list[str] = []
    lock = threading.Lock()

    def publisher():
        k = 1
        while not stop.is_set() and k <= 5:
            rows = [("东闸断面", "氨氮", "2026-11-14 08:00:00", "1.24")]
            rows += [
                (f"站{k}-{i}", "pH", "2026-11-14 08:00:00", "7.3")
                for i in range(k)
            ]
            upload_and_publish(worker, app, rows)
            k += 1

    def reader():
        while not stop.is_set():
            conn = connect(app.config["DATABASE"])
            try:
                snapshot = services.load_published_snapshot(
                    conn, "2026-11-14"
                )
            finally:
                conn.close()
            if snapshot.version is None:
                with lock:
                    problems.append("no published version visible")
                continue
            if snapshot.error:
                with lock:
                    problems.append(snapshot.error)
                continue
            if len(snapshot.rows) != snapshot.version["row_count"]:
                with lock:
                    problems.append(
                        "mixed detail: v%d says %d rows, read %d"
                        % (
                            snapshot.version["version_no"],
                            snapshot.version["row_count"],
                            len(snapshot.rows),
                        )
                    )
            # All returned rows must belong to the named version.
            version_id = snapshot.version["id"]
            check = connect(app.config["DATABASE"])
            try:
                foreign = check.execute(
                    "SELECT COUNT(*) FROM observations WHERE version_id = ?",
                    (version_id,),
                ).fetchone()[0]
            finally:
                check.close()
            if len(snapshot.rows) != foreign:
                with lock:
                    problems.append(
                        "rows from another version in v%d snapshot"
                        % snapshot.version["version_no"]
                    )

    publish_thread = threading.Thread(target=publisher)
    read_threads = [threading.Thread(target=reader) for _ in range(3)]
    publish_thread.start()
    for thread in read_threads:
        thread.start()
    publish_thread.join()
    stop.set()
    for thread in read_threads:
        thread.join()

    assert problems == []


def test_reconcile_uses_only_latest_published_version_after_supersede(client, app):
    first = upload_and_publish(
        client, app, [("东闸断面", "氨氮", "2026-11-14 08:00:00", "1.24")]
    )
    second = upload_and_publish(
        client,
        app,
        [
            ("东闸断面", "氨氮", "2026-11-14 08:00:00", "1.99"),
            ("东闸断面", "pH", "2026-11-14 08:00:00", "7.31"),
        ],
    )
    assert second["version_no"] == 2
    conn = connect(app.config["DATABASE"])
    try:
        statuses = {
            row["version_no"]: row["status"]
            for row in conn.execute("SELECT * FROM versions")
        }
    finally:
        conn.close()
    assert statuses == {1: "superseded", 2: "published"}

    response = post_retest(
        client,
        [("东闸断面", "氨氮", "2026-11-14 08:00", "1.50")],
        window="30",
        deviation="0.5",
    )
    body = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "v2" in body
    # Deviation measured against v2's 1.99, not v1's 1.24.
    assert "0.49" in body
    # v2 has two rows, so one published-side reading is unmatched.
    assert "已发布侧未配对" in body


def test_retest_upload_does_not_create_a_version(client, app):
    upload_and_publish(
        client, app, [("东闸断面", "氨氮", "2026-11-14 08:00:00", "1.24")]
    )
    before = len(services.list_versions(connect(app.config["DATABASE"])))
    post_retest(client, [("东闸断面", "氨氮", "2026-11-14 08:00", "1.2")])
    conn = connect(app.config["DATABASE"])
    try:
        after = len(services.list_versions(conn))
    finally:
        conn.close()
    assert before == after == 1


def test_retest_download_name_is_sanitised_and_temp_is_removed(client, app):
    upload_and_publish(
        client, app, [("东闸断面", "氨氮", "2026-11-14 08:00:00", "1.24")]
    )
    post_retest(
        client,
        [("东闸断面", "氨氮", "2026-11-14 08:00", "1.2")],
        name="../../evil.csv",
    )
    tmp_dir = Path(app.config["FILES_DIR"]) / ".tmp"
    leftovers = [p for p in tmp_dir.iterdir()] if tmp_dir.exists() else []
    assert leftovers == []
