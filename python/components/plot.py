import pandas as pd
import matplotlib.pyplot as plt
import os


class PlotResults:
    """
    This PlotResults class is used to plot the inference times.
    """

    def __init__(self, input_path: str) -> None:
        """
        Initializes the PlotResults class and plots the inference times.

        Args:
            input_path (str): The path where the inference times CSV file is located.
        """

        # define files
        py_inference_measures = os.path.join(input_path, "inference_times.csv")

        # load CSV data
        data = pd.read_csv(py_inference_measures)

        # extract columns
        passes = data["passes"]
        times = data["time_seconds"]

        # plotting
        plt.figure(figsize=(8, 5))
        plt.plot(
            passes, times, marker="o", linestyle="-", color="b", label="Inference Time"
        )

        plt.xlabel("Number of Passes")
        plt.ylabel("Time (seconds)")
        plt.title("Inference Time vs Number of Passes")
        plt.grid(True)
        plt.legend()
        # log scale to show wide range clearly
        plt.xscale("log")
        plt.tight_layout()
        plt.show()
