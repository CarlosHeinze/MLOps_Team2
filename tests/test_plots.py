import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from team2_mlops2026 import plots  # noqa: E402


def test_grouped_bars_draw_a_bar_per_cell_and_the_references():
    wape = pd.DataFrame(
        {"departures": [0.65, 0.53], "arrivals": [0.60, 0.50]}, index=["baseline", "model"]
    )
    fig = plots.plot_grouped_bars(wape, "Peak WAPE", "WAPE", {"departures": 0.584})
    (ax,) = fig.axes
    assert len(ax.patches) == 4
    assert [t.get_text() for t in ax.get_xticklabels()] == ["baseline", "model"]
    assert [line.get_ydata()[0] for line in ax.get_lines()] == [0.584]
    assert ax.containers[0].datavalues.tolist() == [0.65, 0.53]
    plt.close(fig)
