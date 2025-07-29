from components import Setup, Inference, PlotResults

if __name__ == "__main__":
    output_path = "data/"
    passes_list = [1, 10, 100, 1000, 10000]
    batch_sizes_list = [1, 6, 16, 64]

    print("Setup")
    print("-------------")
    Setup(output_path)

    print("\nInference")
    print("-------------")
    inf = Inference(output_path, passes_list, batch_sizes_list)
    # inf.execute_externel_inference("path")

    print("\nPloting")
    print("------------")
    PlotResults(output_path)
