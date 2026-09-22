"""Evidence figures built with matplotlib's object API (no pyplot state).

Using ``matplotlib.figure.Figure`` directly never switches the caller's backend,
so these helpers are safe inside a notebook that renders figures inline.
Colours are the first two slots of a CVD-validated categorical palette; the
decoder or lattice size is always also given by marker, line style, and legend.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ai_qec.evaluation.scientific.result import (
    ScientificAcceptanceResult,
    ScientificEvaluationResult,
)

SERIES = ("#2a78d6", "#eb6834")
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
if TYPE_CHECKING:
    from matplotlib.figure import Figure

HOMOLOGY_LABELS = {"00": "h0", "10": "h1", "01": "h2", "11": "h3"}


@dataclass(frozen=True, slots=True)
class SweepPoint:
    """One (L, p) point of a paper sweep and its scientific results."""

    distance: int
    physical_error_rate: float
    scientific: ScientificEvaluationResult
    acceptance: ScientificAcceptanceResult | None = None


def _figure(width: float, height: float, rows: int = 1, columns: int = 1):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height), layout="constrained", facecolor=SURFACE)
    axes = figure.subplots(rows, columns, squeeze=False)
    for axis in axes.flat:
        axis.set_facecolor(SURFACE)
        axis.grid(color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(MUTED)
        axis.tick_params(colors=MUTED, labelcolor=INK)
    return figure, axes


def save_figure(figure: "Figure", path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    return path


def figure_png(figure: "Figure", dpi: int = 150) -> bytes:
    """PNG bytes for inline display without depending on the notebook's matplotlib backend."""

    import io

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, facecolor=SURFACE)
    return buffer.getvalue()


def _class_order(counts: Mapping[str, int]) -> list[str]:
    return sorted(counts, key=lambda label: HOMOLOGY_LABELS.get(label, label))


def _short(decoder_id: str) -> str:
    return {"joint-error-syndrome-rbm": "RBM (NeD)", "pymatching-cpu-decoder": "MWPM"}.get(
        decoder_id, decoder_id
    )


def plot_training_history(history: Sequence[Mapping[str, object]]) -> "Figure":
    figure, axes = _figure(6.0, 3.4)
    axis = axes[0, 0]
    epochs = [row["epoch"] for row in history]
    axis.plot(
        epochs,
        [row["train_reconstruction_bce"] for row in history],
        color=SERIES[0],
        linewidth=2,
        marker="o",
        markersize=4,
        label="train (monitor subset)",
    )
    axis.plot(
        epochs,
        [row["validation_reconstruction_bce"] for row in history],
        color=SERIES[1],
        linewidth=2,
        linestyle="--",
        marker="s",
        markersize=4,
        label="validation",
    )
    axis.set_xlabel("epoch")
    axis.set_ylabel("one-step reconstruction BCE")
    axis.set_title("RBM training monitor (not used for model selection)", color=INK, fontsize=10)
    axis.legend(frameon=False)
    return figure


def plot_decoder_comparison(
    result: ScientificEvaluationResult, acceptance: ScientificAcceptanceResult | None = None
) -> "Figure":
    figure, axes = _figure(5.2, 3.6)
    axis = axes[0, 0]
    names = [_short(item.decoder_id) for item in result.decoder_results]
    rates = [item.logical_error_rate.value for item in result.decoder_results]
    lower = [
        item.logical_error_rate.value - item.logical_error_rate.interval_low
        for item in result.decoder_results
    ]
    upper = [
        item.logical_error_rate.interval_high - item.logical_error_rate.value
        for item in result.decoder_results
    ]
    positions = range(len(names))
    axis.bar(
        positions, rates, width=0.55, color=SERIES[: len(names)], edgecolor=SURFACE, linewidth=2
    )
    axis.errorbar(
        positions, rates, yerr=[lower, upper], fmt="none", ecolor=INK, capsize=5, linewidth=1.2
    )
    for position, item in zip(positions, result.decoder_results):
        estimate = item.logical_error_rate
        axis.annotate(
            f"{estimate.value:.3f}\n({estimate.numerator}/{estimate.denominator})",
            (position, estimate.interval_high),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            va="bottom",
            color=INK,
            fontsize=8,
        )
    axis.set_xticks(list(positions), names)
    axis.set_ylabel(r"$P_{fail}$ (Wilson 95% CI)")
    axis.set_ylim(
        0,
        min(
            1.0,
            max(item.logical_error_rate.interval_high for item in result.decoder_results) * 1.3
            + 0.02,
        ),
    )
    title = "Logical failure rate on the same test shots"
    if acceptance is not None:
        title += f"\nAccuracy Gate {acceptance.gate_id}: {acceptance.decision.value.upper()}"
    axis.set_title(title, color=INK, fontsize=10)
    return figure


def plot_logical_classes(result: ScientificEvaluationResult) -> "Figure":
    figure, axes = _figure(5.6, 3.6)
    axis = axes[0, 0]
    labels = _class_order(result.decoder_results[0].logical_class_counts)
    width = 0.8 / len(result.decoder_results)
    for slot, item in enumerate(result.decoder_results):
        total = sum(item.logical_class_counts.values()) or 1
        offsets = [
            index + (slot - (len(result.decoder_results) - 1) / 2) * width
            for index in range(len(labels))
        ]
        axis.bar(
            offsets,
            [100 * item.logical_class_counts[label] / total for label in labels],
            width=width * 0.92,
            color=SERIES[slot % len(SERIES)],
            edgecolor=SURFACE,
            linewidth=2,
            hatch=None if slot == 0 else "//",
            label=_short(item.decoder_id),
        )
    axis.set_xticks(range(len(labels)), [HOMOLOGY_LABELS.get(label, label) for label in labels])
    axis.set_xlabel("homology class of e ⊕ r (h0 = success)")
    axis.set_ylabel("% of accepted recoveries")
    axis.set_ylim(0, 100)
    axis.legend(frameon=False)
    return figure


def plot_failure_rate_sweep(points: Sequence[SweepPoint]) -> "Figure":
    """Paper Fig. 3: P_fail versus p; lines = MWPM, markers = neural decoder."""

    figure, axes = _figure(6.4, 4.4)
    axis = axes[0, 0]
    distances = sorted({point.distance for point in points})
    markers = ("^", "s", "o", "D")
    for slot, distance in enumerate(distances):
        color = SERIES[slot % len(SERIES)]
        selected = sorted(
            (point for point in points if point.distance == distance),
            key=lambda point: point.physical_error_rate,
        )
        p = [point.physical_error_rate for point in selected]
        candidate = [point.scientific.decoder_results[0].logical_error_rate for point in selected]
        baseline = [point.scientific.decoder_results[1].logical_error_rate for point in selected]
        axis.plot(
            p,
            [item.value for item in baseline],
            color=color,
            linewidth=2,
            label=f"L = {distance}, {_short(selected[0].scientific.decoder_results[1].decoder_id)}",
        )
        axis.errorbar(
            p,
            [item.value for item in candidate],
            yerr=[
                [item.value - item.interval_low for item in candidate],
                [item.interval_high - item.value for item in candidate],
            ],
            fmt=markers[slot % len(markers)],
            markersize=7,
            color=color,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            ecolor=color,
            elinewidth=1,
            capsize=3,
            linestyle="none",
            label=f"L = {distance}, {_short(selected[0].scientific.decoder_results[0].decoder_id)}",
        )
    axis.set_xlabel(r"$p_{err}$")
    axis.set_ylabel(r"$P_{fail}$")
    axis.set_title("Logical failure probability (markers: 95% Wilson CI)", color=INK, fontsize=10)
    axis.legend(frameon=False, loc="upper left")
    return figure


def plot_logical_class_histograms(points: Sequence[SweepPoint], decoder_index: int = 0) -> "Figure":
    """Paper Fig. 4: homology classes of e ⊕ r returned by one decoder at several p."""

    points = sorted(points, key=lambda point: point.physical_error_rate)
    columns = 2 if len(points) > 1 else 1
    rows = (len(points) + columns - 1) // columns
    figure, axes = _figure(3.2 * columns, 2.6 * rows, rows, columns)
    for axis, point in zip(axes.flat, points):
        item = point.scientific.decoder_results[decoder_index]
        labels = _class_order(item.logical_class_counts)
        total = sum(item.logical_class_counts.values()) or 1
        shares = [100 * item.logical_class_counts[label] / total for label in labels]
        axis.bar(
            range(len(labels)),
            shares,
            width=0.6,
            color=[SERIES[0]] + [SERIES[1]] * (len(labels) - 1),
            edgecolor=SURFACE,
            linewidth=2,
        )
        for index, share in enumerate(shares):
            axis.annotate(
                f"{share:.0f}%",
                (index, share),
                textcoords="offset points",
                xytext=(0, 2),
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK,
            )
        axis.set_xticks(range(len(labels)), [HOMOLOGY_LABELS.get(label, label) for label in labels])
        axis.set_ylim(0, 105)
        axis.set_title(
            f"L = {point.distance}, $p_{{err}}$ = {point.physical_error_rate:.2f}",
            color=INK,
            fontsize=10,
        )
    for axis in list(axes.flat)[len(points) :]:
        axis.set_visible(False)
    figure.suptitle(
        f"{_short(points[0].scientific.decoder_results[decoder_index].decoder_id)}: "
        "% of accepted recoveries per homology class (blue: h0, trivial)",
        color=INK,
        fontsize=10,
    )
    return figure
