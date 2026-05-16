from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False


def save_fig(path: str, dpi: int = 150) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.show()


def bar_h(
    series: pd.Series,
    title: str,
    xlabel: str,
    color_map: dict | None = None,
    figsize: tuple = (10, 6),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    colors = [color_map.get(i, "#4C72B0") for i in series.index] if color_map else "#4C72B0"
    series.plot.barh(ax=ax, color=colors)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("피처")
    ax.invert_yaxis()
