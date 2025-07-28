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


class Inference:
    """
    This Inference class runs the inference on the model.

    Methodes:
        __init__: Initializes the inference, loads the model and dataset, and runs inference
        load_iris_data: Loads the iris dataset from a CSV file
        run_inference: Runs inference on the model for a specified number of passes
        execute_externel_inference: Executes an external inference program
    """

    def __init__(self, output_path: str) -> None:
        """
        Performs the inference.

        Args:
            output_path (str): The output path.
        """

        csv_path = os.path.join(output_path, "iris.csv")
        model_path = os.path.join(output_path, "model_state_dict.pt")
        inference_output_path = os.path.join(output_path, "inference_times.csv")
        batch_size = 6
        passes_list = [1, 10, 100, 1000, 10000]

        # layer sizes
        b = 4  # Iris features
        c = 64
        d = 16
        e = 3  # Iris species

        # Load data and model
        print("Load model and dataset...")
        dataset = self.load_iris_data(csv_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = model = BasicNet(b, c, d, e)
        model.load_state_dict(torch.load(model_path))
        # use only CPU
        model.to(torch.device("cpu"))
        model.eval()

        # Time inference
        print("Run inference on CPU...")
        results = []
        for passes in passes_list:
            print(f"For {passes} passes:")
            duration = self.run_inference(model, dataloader, passes)
            print(f"{passes} passes took {duration:.4f} seconds")
            results.append({"passes": passes, "time_seconds": duration})

        # Save results to CSV
        print("Save Results...")
        results_df = pd.DataFrame(results)
        results_df.to_csv(inference_output_path, index=False)
        print("Done.")

    def load_iris_data(self, csv_path: str) -> None:
        """
        Loads the iris dataset from a CSV file.

        Args:
            csv_path (str): The path to the CSV file.
        """

        df = pd.read_csv(csv_path)
        X = df.iloc[:, :-1].values
        y = LabelEncoder().fit_transform(df.iloc[:, -1])  # Encode species labels
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
            for i in tqdm(range(passes)):
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
            # Windows
            subprocess.run(["my_program.exe"])
        else:
            # Unix/Linux/macOS
            subprocess.run(["./my_program"])
