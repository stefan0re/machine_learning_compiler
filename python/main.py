from components import Setup, Inference, PlotResults
from tests import Tests
import os

if __name__ == "__main__":
    output_path = "data/"
    passes_list = [1, 10, 100, 1000, 10000]
    batch_sizes_list = [1, 6, 16, 64]

    dataset_path = os.path.join(output_path, "iris.csv")
    model_path = os.path.join(output_path, "model_state_dict.pt")

    print("Setup")
    print("----------------------")
    # Setup(output_path)

    print("\nInference")
    print("----------------------")
    # inf = Inference(output_path, passes_list, batch_sizes_list)
    # inf.execute_externel_inference("path")

    print("\nPloting")
    print("----------------------")
    # PlotResults(output_path)

    print("\nGenerate Example")
    print("----------------------")
    Tests(dataset_path, model_path, output_path)
