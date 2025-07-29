import torch
from modules import BasicNet
import pandas as pd
import os
import numpy as np


class Tests:
    """
    # Get a input sample and calculate the output from ouer network to use in the cpp tests
    """

    def __init__(self, dataset_path: str, model_path: str, output_path: str) -> None:
        """
        Initializes the test class and runs the example generation.

        Args:
            dataset_path (str): Path to the dataset CSV file.
            model_path (str): Path to the trained model state dictionary.
            output_path (str): Path where the example will be saved.
        """

        # model parameters
        a = 1
        b = 4
        c = 64
        d = 16
        e = 3

        # load one sample from the data
        print("Get a sample...")
        df = pd.read_csv(dataset_path)
        # TODO: Inserte batch size here
        X = df.sample(1)
        # filter the class out
        X_numeric = X.select_dtypes(include=[np.number])
        X_tensor = torch.tensor(X_numeric.values, dtype=torch.float32)

        # load the network
        print("Load model...")
        model = BasicNet(b, c, d, e)
        model.load_state_dict(torch.load(model_path))
        model.to(torch.device("cpu"))
        model.eval()

        # run network with that sample
        print("Run model...")
        output = model(X_tensor)

        path = os.path.join(output_path, "example.csv")

        # save input and output
        print("Save output...")
        with open(path, "w") as f:
            # convert tensors to lists and write to file
            f.write(",".join(map(str, X_tensor.numpy().flatten())) + "\n")
            f.write(",".join(map(str, output.detach().numpy().flatten())) + "\n")
