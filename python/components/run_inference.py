import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from modules import BasicNet
import os
from tqdm import tqdm
import subprocess
import sys
from typing import List


class Inference:
    """
    This Inference class runs the inference on the model.

    Methods:
        __init__: Initializes the inference, loads the model and dataset, and runs inference
        load_iris_data: Loads the iris dataset from a CSV file
        run_inference: Runs inference on the model for a specified number of passes
        execute_externel_inference: Executes an external inference program
    """

    def __init__(
        self, output_path: str, passes_list: List, batch_sizes_list: List
    ) -> None:
        """
        Performs the inference.

        Args:
            output_path (str): The output path.
            passes_list (List): List of passes to run inference.
            batch_sizes_list (List): List of batch sizes to use for inference.
        """

        csv_path = os.path.join(output_path, "iris.csv")
        model_path = os.path.join(output_path, "model_state_dict.pt")
        inference_output_path = os.path.join(output_path, "inference_times.csv")

        # layer sizes
        b = 4  # Iris features
        c = 64
        d = 16
        e = 3  # Iris species

        # load data and model
        print("Load model and dataset...")
        dataset = self.load_iris_data(csv_path)

        model = BasicNet(b, c, d, e)
        model.load_state_dict(torch.load(model_path))
        model.to(torch.device("cpu"))
        model.eval()

        # time inference
        print("Run inference on CPU...")
        timing_matrix = {}

        for batch_size in batch_sizes_list:
            print(f"\nBatch size: {batch_size}")
            # for the batch size to be variable, the dataloader has to be new created
            # ech time
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            row = []
            for passes in passes_list:
                print(f"  Passes: {passes}")
                duration = self.run_inference(model, dataloader, passes)
                print(f"    Time: {duration:.4f} seconds")
                row.append(duration)
            timing_matrix[batch_size] = row

        # save results in matrix format
        # transpose so batch sizes are rows
        results_df = pd.DataFrame(timing_matrix, index=passes_list).T
        results_df.index.name = "batch_size"
        results_df.columns = [f"passes_{p}" for p in passes_list]
        results_df.to_csv(inference_output_path)

        print("\nSaved results to:", inference_output_path)

    def load_iris_data(self, csv_path: str) -> TensorDataset:
        """
        Loads the iris dataset from a CSV file.

        Args:
            csv_path (str): The path to the CSV file.
        """

        df = pd.read_csv(csv_path)
        X = df.iloc[:, :-1].values
        # encode species labels
        y = LabelEncoder().fit_transform(df.iloc[:, -1])
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return TensorDataset(X_tensor, y_tensor)

    def run_inference(
        self, model: BasicNet, dataloader: DataLoader, passes: int
    ) -> float:
        """
        Runs inference on the model for a specified number of passes.

        Args:
            model (BasicNet): The model to run inference on.
            dataloader (DataLoader): The DataLoader for the dataset.
            passes (int): The number of passes to run.
        """

        start = time.time()
        with torch.no_grad():
            for _ in tqdm(range(passes), leave=False):
                for inputs, _ in dataloader:
                    _ = model(inputs)
        end = time.time()
        return end - start

    def execute_externel_inference(executable_path: str) -> None:
        """
        Executes an external inference program.

        Args:
            executable_path (str): The path to the external executable.
        """

        if sys.platform.startswith("win"):
            subprocess.run(["my_program.exe"])
        else:
            subprocess.run(["./my_program"])
