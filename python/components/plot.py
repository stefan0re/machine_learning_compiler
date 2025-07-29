import pandas as pd
import matplotlib.pyplot as plt
import os
import math


class PlotResults:
    """
    This PlotResults class is used to plot inference times for each batch size,
    all in a single figure with multiple subplots.
    """

    def __init__(self, input_path: str) -> None:
        """
        Initializes the PlotResults class and generates subplots for each batch size.

        Args:
            input_path (str): The path where the inference_times.csv file is located.
        """

        csv_path = os.path.join(input_path, "inference_times.csv")

        # load matrix-form CSV
        df = pd.read_csv(csv_path, index_col="batch_size")

        # extract passes from column names (e.g., 'passes_1' → 1)
        passes = [int(col.split("_")[1]) for col in df.columns]

        # get the global max time for equal y-axis scaling
        overall_max_time = df.max().max()

        # determine layout of subplots
        num_plots = len(df.index)
        # number of columns in the subplot grid
        cols = 2
        rows = math.ceil(num_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        # make it iterable even if rows=1
        axes = axes.flatten()

        for idx, batch_size in enumerate(df.index):
            times = df.loc[batch_size].values
            ax = axes[idx]

            ax.plot(
                passes,
                times,
                marker="o",
                linestyle="-",
                label=f"Batch Size {batch_size}",
            )

            # annotate each point with its value
            for x, y in zip(passes, times):
                ax.annotate(
                    f"{y:.4f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=8,
                )

            ax.set_title(f"Batch Size {batch_size}")
            ax.set_xlabel("Number of Passes")
            ax.set_ylabel("Time (seconds)")
            ax.set_xscale("log")
            # added slight padding for visibility
            ax.set_ylim(0, overall_max_time * 1.05)
            ax.grid(True)
            ax.legend()

        # hide unused subplots
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
