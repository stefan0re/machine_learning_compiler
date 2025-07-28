from components import Setup, Inference, PlotResults

if __name__ == "__main__":
    output_path = "data/"
    print("Setup")
    print("-------------")
    Setup(output_path)

    print("\nInference")
    print("-------------")
    inf = Inference(output_path)
    # inf.execute_externel_inference("path")

    print("\nPloting")
    print("------------")
    PlotResults(output_path)
