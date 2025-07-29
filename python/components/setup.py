import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from typing import Tuple, List
from modules import BasicNet
import os


class Setup:
    """
    This Setup function does the presteps for the inference.

    Methods:
        __init__: Initializes the setup, loads the dataset, trains the model, and saves it
        load_and_save_dataset: Loads the iris dataset and saves it to a CSV file and return the data
        save_cpp_model: Saves the model weights and biases to a file for C++ model loading
        test_model: Evaluates the model's accuracy and throughput

    """

    def __init__(self, output_path: str) -> None:
        """
        Initializes the setup, loads the dataset, trains the model, and saves it

        Args:
            output_path (str): The path where to output the files.
        """

        # Model setup
        a = 1  # batch size
        b = 4  # Iris features
        c = 64
        d = 16
        e = 3  # Iris species

        # define output paths
        iris_output_path = os.path.join(output_path, "iris.csv")
        cpp_model_path = os.path.join(output_path, "model.torchpp")
        model_output_path = os.path.join(output_path, "model_state_dict.pt")
        print("Outputting to: ")
        print(f"\t{iris_output_path}")
        print(f"\t{cpp_model_path}")

        print("Load and Save dataset...")
        X_train, X_test, y_train, y_test = self.load_and_save_dataset(iris_output_path)

        print("Setup model...")
        model = BasicNet(b, c, d, e)
        # use only CPU
        model.to(torch.device("cpu"))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # training loop
        print("Train Model...")
        epochs = 100
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        # save the model
        print("Save model...")
        torch.save(model.state_dict(), model_output_path)

        self.save_cpp_model(model, cpp_model_path)

        # evaluate model
        print("Evaluating the model...")
        self.test_model(model, X_test, y_test)

        # print the einsum-notation for the model
        print("Model details: ")
        print(f"a: {a}")
        print(f"b: {b}")
        print(f"c: {c}")
        print(f"d: {d}")
        print(f"e: {e}")
        # match standard PyTorch format (batch-first)
        print(f"[[[a,b], [c,b] -> [a,c]], [d,c] -> [a,d]], [e,d] -> [a,e]]")
        print("Done.")

    def load_and_save_dataset(self, output_path: str) -> Tuple[List, List, List, List]:
        """
        Load the iris dataset and 1. save it to a .csv file and 2. return the data

        Args:
            output_path(str): The path where the dataset should be outputed.
        """

        # load and preprocess the Iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target

        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # convert to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Export to csv:

        # create a DataFrame with the data and feature names
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

        # Add the target column (species)
        df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

        # Save to CSV
        df.to_csv(output_path, index=False)

        return X_train, X_test, y_train, y_test

    # save weights and biases to file
    def save_cpp_model(self, model: BasicNet, filename: str) -> None:
        """
        Save the weights and biases to a file for the cpp model to load from.

        Args:
            model (BasicNet): The model that should be saved.
            filename (str): The output path.
        """

        with open(filename, "w") as f:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # row-major (C-style) order
                    # Tensor(out_features, in_features)
                    # -> w[0,0], w[0,1], ..., w[0,N-1], w[1,0], ..., w[O-1,N-1]
                    # with O = out_features, N = in_features
                    flat = param.detach().numpy().flatten()
                    line = ",".join(str(x) for x in flat)
                    f.write(line + "\n")

    def test_model(self, model: BasicNet, X_test: List, y_test: List) -> None:
        """
        Evaluates the model and output its accuracy and throughput.

        Args:
            model (BasicNet): The model that should be evaluated.
            X_test (List): The input test values.
            y_test (List): The corresponding output values.
        """

        # Evaluation mode
        model.eval()

        # Disable gradient computation (faster & safer for inference)
        with torch.no_grad():
            outputs = model(X_test)

            # accuracy
            _, predicted = torch.max(outputs, 1)  # Get class with highest score
            accuracy = (predicted == y_test).sum().item() / len(y_test)
            print(f"Test Accuracy: {accuracy * 100:.2f}%")

            # throughput
            start = time.perf_counter()
            for _ in range(10000):
                _ = model(X_test)
            end = time.perf_counter()
            exec_time = end - start
            print(f"Model evaluation took {exec_time:.8f} seconds")
            flops = 2 * 10000 * 4 * 64 * 16 * 3 - 3
            print(f"FLOPs: {(flops / exec_time) * 1e-9:,}")
