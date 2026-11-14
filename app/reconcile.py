"""One-to-one reconciliation of laboratory retest readings.

Retest readings are paired with the observations of a *single* published
version.  Pairing is scoped to the same site and metric and only connects
readings whose sampling times differ by at most the configured window.  Among
all such pairings it first maximises the number of pairs and then minimises
the total time difference.  Every reading is used at most once.

The maximum absolute value deviation is a quality-control threshold: it does
not restrict which readings may pair, but pairs whose deviation exceeds it
are reported separately.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from typing import Any, Iterable, Sequence

MICROSECONDS_PER_MINUTE = 60_000_000
MICROSECONDS_PER_SECOND = 1_000_000

# A dense site/metric group can produce N x M dynamic-programming cells.
# Refuse rather than exhaust memory; the reviewer can narrow the time window.
MAX_CANDIDATE_PAIRS = 1_000_000


class ParameterError(ValueError):
    """Raised when a reviewer-supplied parameter is invalid."""


@dataclass(frozen=True)
class RetestReading:
    row_no: int
    site: str
    metric: str
    observed_at: str
    value_text: str
    value: Decimal
    time_micros: int


@dataclass(frozen=True)
class PublishedReading:
    row_no: int
    site: str
    metric: str
    observed_at: str
    value_text: str
    value: Decimal
    time_micros: int


@dataclass(frozen=True)
class Pair:
    retest: RetestReading
    published: PublishedReading
    time_diff_micros: int
    deviation: Decimal
    deviation_limit: Decimal

    @property
    def exceeds_deviation(self) -> bool:
        return self.deviation > self.deviation_limit

    @property
    def time_diff_minutes_text(self) -> str:
        return _format_units(self.time_diff_micros, MICROSECONDS_PER_MINUTE)

    @property
    def time_diff_seconds_text(self) -> str:
        return _format_units(self.time_diff_micros, MICROSECONDS_PER_SECOND)

    @property
    def time_diff_text(self) -> str:
        """Human-readable difference, e.g. ``1 分 30 秒`` or ``45 秒``."""
        total_seconds, remainder_micros = divmod(
            self.time_diff_micros, MICROSECONDS_PER_SECOND
        )
        minutes, seconds = divmod(total_seconds, 60)
        parts: list[str] = []
        if minutes:
            parts.append(f"{minutes} 分")
        if seconds or remainder_micros or not parts:
            fraction = ""
            if remainder_micros:
                fraction = "." + _canonical_decimal(
                    Decimal(remainder_micros) / MICROSECONDS_PER_SECOND
                ).split(".")[-1]
            parts.append(f"{seconds}{fraction} 秒")
        return " ".join(parts)

    @property
    def deviation_text(self) -> str:
        return _canonical_decimal(self.deviation)

    @property
    def excess_text(self) -> str:
        """How far the deviation exceeds the threshold (never negative)."""
        return _canonical_decimal(self.deviation - self.deviation_limit)


@dataclass
class ReconcileReport:
    pairs: list[Pair]
    over_limit: list[Pair]
    retest_unmatched: list[tuple[RetestReading, str]]
    published_unmatched: list[tuple[PublishedReading, str]]
    retest_total: int
    published_total: int
    max_time_diff_minutes: Decimal
    max_abs_deviation: Decimal

    @property
    def pair_count(self) -> int:
        return len(self.pairs)

    @property
    def total_time_diff_micros(self) -> int:
        return sum(pair.time_diff_micros for pair in self.pairs)

    @property
    def total_time_diff_text(self) -> str:
        return _format_units(self.total_time_diff_micros, MICROSECONDS_PER_MINUTE)


# ---------------------------------------------------------------------------
# Parameter parsing
# ---------------------------------------------------------------------------


MAX_TIME_WINDOW_MINUTES = Decimal(1440)  # one day; a larger window is meaningless


def parse_time_window_minutes(raw: str | None) -> tuple[Decimal, int]:
    """Return (minutes, window_in_microseconds) for a reviewer input."""
    value = _parse_non_negative_decimal(raw, "最大时间差（分钟）")
    if value > MAX_TIME_WINDOW_MINUTES:
        raise ParameterError("最大时间差不能超过 1440 分钟（一天）。")
    micros = int(
        (value * MICROSECONDS_PER_MINUTE).to_integral_value(rounding=ROUND_FLOOR)
    )
    return value, micros


def parse_deviation(raw: str | None) -> Decimal:
    value = _parse_non_negative_decimal(raw, "最大绝对数值偏差")
    if value.adjusted() > 30:
        raise ParameterError("最大绝对数值偏差过大")
    return value


def _parse_non_negative_decimal(raw: str | None, label: str) -> Decimal:
    text = (raw or "").strip()
    if not text:
        raise ParameterError(f"请填写{label}。")
    if len(text) > 64:
        raise ParameterError(f"{label}过长。")
    try:
        value = Decimal(text)
    except (InvalidOperation, ValueError):
        raise ParameterError(f"{label}必须是数字。") from None
    if not value.is_finite():
        raise ParameterError(f"{label}必须是有限数字。")
    if value < 0:
        raise ParameterError(f"{label}不能为负数。")
    return value


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def parse_time_micros(value: str) -> int:
    parsed = datetime.fromisoformat(value)
    return _datetime_to_micros(parsed)


def _datetime_to_micros(value: datetime) -> int:
    seconds = (
        (value.toordinal() * 24 + value.hour) * 3600
        + value.minute * 60
        + value.second
    )
    return seconds * MICROSECONDS_PER_SECOND + value.microsecond


def _format_units(micros: int, unit_micros: int) -> str:
    if unit_micros <= 0:
        return "0"
    return _canonical_decimal(Decimal(micros) / Decimal(unit_micros))


def _canonical_decimal(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in ("", "-0"):
        return "0"
    return text


# ---------------------------------------------------------------------------
# Maximum-cardinality, minimum-total-time matching
# ---------------------------------------------------------------------------


def _best_pairing(
    retest: Sequence[RetestReading],
    published: Sequence[PublishedReading],
    window_micros: int,
) -> list[tuple[int, int]]:
    """Pair indices one-to-one: max cardinality first, then min total cost.

    Both sides are sorted by sampling time.  For |time difference| costs, any
    crossed pairing ``(i -> j')`` and ``(i' -> j)`` (``i < i'``, ``j < j'``)
    can be uncrossed into ``(i -> j)`` and ``(i' -> j')`` without reducing the
    number of pairs, increasing the total time difference, or breaking the
    maximum-time-difference window.  An optimal pairing therefore exists that
    never crosses, so a prefix dynamic program over the two sorted lists finds
    a lexicographic optimum: maximise pairs, then minimise total difference.

    The DP value packs both objectives into one integer: every pair contributes
    ``scale - cost`` with ``scale`` larger than any achievable total cost, so
    more pairs always outranks any saving in time.
    """
    n_retest = len(retest)
    n_published = len(published)
    if n_retest == 0 or n_published == 0:
        return []
    if (n_retest + 1) * (n_published + 1) > MAX_CANDIDATE_PAIRS:
        raise ParameterError(
            "同一站点与指标的候选配对数过多（超过 %d），请缩小最大时间差后重试。"
            % MAX_CANDIDATE_PAIRS
        )

    min_pairs = min(n_retest, n_published)
    scale = min_pairs * window_micros + 1
    previous = [0] * (n_published + 1)
    # Keep the full table only as wide as needed for reconstruction: one
    # compact int64 row per retest reading, ~8 bytes per cell.
    rows = array("q", [0]) * (n_retest + 1)
    width = n_published + 1
    rows[0:width] = array("q", previous)
    for i in range(1, n_retest + 1):
        left = retest[i - 1]
        current = [0] * width
        for j in range(1, width):
            best = previous[j]
            if current[j - 1] > best:
                best = current[j - 1]
            delta = abs(left.time_micros - published[j - 1].time_micros)
            if delta <= window_micros:
                paired = previous[j - 1] + scale - delta
                if paired > best:
                    best = paired
            current[j] = best
        rows[i * width : (i + 1) * width] = array("q", current)
        previous = current

    def cell(i: int, j: int) -> int:
        return rows[i * width + j]

    # Reconstruct: prefer skipping a reading when that keeps the optimum.
    pairs: list[tuple[int, int]] = []
    i, j = n_retest, n_published
    while i > 0 and j > 0:
        value = cell(i, j)
        if value == cell(i, j - 1):
            j -= 1
            continue
        if value == cell(i - 1, j):
            i -= 1
            continue
        pairs.append((i - 1, j - 1))
        i -= 1
        j -= 1
    pairs.reverse()
    return pairs


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def retest_readings_from_observations(observations: Iterable[Any]) -> list[RetestReading]:
    readings: list[RetestReading] = []
    for item in observations:
        readings.append(
            RetestReading(
                row_no=item.row_no,
                site=item.site,
                metric=item.metric,
                observed_at=item.observed_at,
                value_text=item.value_text,
                value=Decimal(item.value_text),
                time_micros=parse_time_micros(item.observed_at),
            )
        )
    return readings


def published_readings_from_rows(rows: Iterable[Any]) -> list[PublishedReading]:
    readings: list[PublishedReading] = []
    for row in rows:
        readings.append(
            PublishedReading(
                row_no=int(row["row_no"]),
                site=row["site"],
                metric=row["metric"],
                observed_at=row["observed_at"],
                value_text=row["value_text"],
                value=Decimal(row["value_text"]),
                time_micros=parse_time_micros(row["observed_at"]),
            )
        )
    return readings


def reconcile(
    retest: Sequence[RetestReading],
    published: Sequence[PublishedReading],
    *,
    window_micros: int,
    max_abs_deviation: Decimal,
    max_time_diff_minutes: Decimal,
) -> ReconcileReport:
    # Group positions by (site, metric): pairing is only ever considered
    # within one group, so a reading can never be reused across groups.
    retest_groups: dict[tuple[str, str], list[int]] = {}
    for position, reading in enumerate(retest):
        retest_groups.setdefault((reading.site, reading.metric), []).append(position)

    published_groups: dict[tuple[str, str], list[int]] = {}
    for position, reading in enumerate(published):
        published_groups.setdefault((reading.site, reading.metric), []).append(position)

    pairs: list[Pair] = []
    matched_retest: set[int] = set()
    matched_published: set[int] = set()

    for key in retest_groups.keys() | published_groups.keys():
        retest_positions = sorted(
            retest_groups.get(key, []),
            key=lambda position: (retest[position].observed_at, retest[position].row_no),
        )
        published_positions = sorted(
            published_groups.get(key, []),
            key=lambda position: (
                published[position].observed_at,
                published[position].row_no,
            ),
        )
        group_retest = [retest[position] for position in retest_positions]
        group_published = [published[position] for position in published_positions]
        for i, j in _best_pairing(group_retest, group_published, window_micros):
            left = group_retest[i]
            right = group_published[j]
            matched_retest.add(retest_positions[i])
            matched_published.add(published_positions[j])
            deviation = abs(left.value - right.value)
            pairs.append(
                Pair(
                    retest=left,
                    published=right,
                    time_diff_micros=abs(left.time_micros - right.time_micros),
                    deviation=deviation,
                    deviation_limit=max_abs_deviation,
                )
            )

    pairs.sort(
        key=lambda pair: (
            pair.retest.site,
            pair.retest.metric,
            pair.retest.observed_at,
            pair.retest.row_no,
            pair.published.observed_at,
        )
    )
    over_limit = [pair for pair in pairs if pair.exceeds_deviation]

    retest_unmatched = [
        (
            reading,
            _retest_unmatched_reason(
                reading,
                [published[position] for position in published_groups.get(
                    (reading.site, reading.metric), []
                )],
                window_micros,
            ),
        )
        for position, reading in sorted(
            enumerate(retest), key=lambda item: item[1].row_no
        )
        if position not in matched_retest
    ]
    published_unmatched = [
        (
            reading,
            _published_unmatched_reason(
                reading,
                [retest[position] for position in retest_groups.get(
                    (reading.site, reading.metric), []
                )],
                window_micros,
            ),
        )
        for position, reading in sorted(
            enumerate(published),
            key=lambda item: (
                item[1].site,
                item[1].metric,
                item[1].observed_at,
                item[1].row_no,
            ),
        )
        if position not in matched_published
    ]

    return ReconcileReport(
        pairs=pairs,
        over_limit=over_limit,
        retest_unmatched=retest_unmatched,
        published_unmatched=published_unmatched,
        retest_total=len(retest),
        published_total=len(published),
        max_time_diff_minutes=max_time_diff_minutes,
        max_abs_deviation=max_abs_deviation,
    )


def _retest_unmatched_reason(
    reading: RetestReading,
    counterparts: Sequence[PublishedReading],
    window_micros: int,
) -> str:
    if any(
        abs(reading.time_micros - other.time_micros) <= window_micros
        for other in counterparts
    ):
        return "时间差阈值内的已发布读数均已被其他复测读数配对"
    return "该日期已发布明细中无同站同指标、且时间差在阈值内的读数"


def _published_unmatched_reason(
    reading: PublishedReading,
    counterparts: Sequence[RetestReading],
    window_micros: int,
) -> str:
    if any(
        abs(reading.time_micros - other.time_micros) <= window_micros
        for other in counterparts
    ):
        return "时间差阈值内的复测读数均已被其他已发布读数配对"
    return "复测文件中无同站同指标、且时间差在阈值内的读数"
