Individual Phase
=======================================

Overview
--------

Machine learning is a rapidly growing field of computer science. Neural networks are used extensively in many modern applications. Crucial to their success is not only the accuracy of the models but also the speed at which they can perform predictions. While fast training is desirable, real-time prediction is essential for tasks such as self-driving cars, medical imaging, security systems, and more.

Popular frameworks like TensorFlow and PyTorch enable developers to build and maintain large networks easily with minimal boilerplate code. However, these libraries sometimes lack hardware-specific optimizations, particularly when it comes to real-time inference, leaving room for potential performance improvements.

This is why we have chosen to use our custom tensor contraction library, specifically our optimized einsum implementation, to evaluate whether our hardware-accelerated tensor multiplications can outperform PyTorch’s forward pass in prediction tasks.

Since prediction time is more critical than training time in most real-world applications, we will first train a model using PyTorch's robust tools. We will then benchmark the inference performance between PyTorch and our implementation. This involves implementing the necessary components to load a pre-trained PyTorch model, process a dataset, perform the forward pass and output the predictions in a human-readable format.

Feature Outline
----------------

- Dataset generation and model training
- Inference benchmarking
- Visualization of results
- Validation using a C++ implementation

Run the full workflow with:

.. code-block:: bash

    python main.py

Folder Structure
----------------

.. code-block:: text

    python/
    ├── components/
    │   ├── __init__.py
    │   ├── plot.py              # Contains PlotResults class
    │   ├── run_inference.py     # Contains Inference class
    │   └── setup.py             # Contains Setup class
    ├── data/
    │   ├── example.csv
    │   ├── inference_times.csv
    │   ├── iris.csv
    │   ├── model_state_dict.pt
    │   └── model.torchpp
    ├── main.py                  # Entrypoint to run the pipeline
    ├── modules/
    │   ├── __init__.py
    │   └── basic_net.py         # PyTorch model definition
    ├── requierments.txt
    └── tests/
        ├── __init__.py
        ├── sample.py            # Contains Tests class
    test/
    └── basic_net/
        └── correct_calculations.cpp  # C++ validator using Tensor class

Components
----------

**1. Setup (`components/setup.py`)**

- Loads the Iris dataset using `sklearn`.
- Saves it to `data/iris.csv`.
- Trains a PyTorch model defined in `modules/basic_net.py`.
- Saves the model in two formats:
    - `model_state_dict.pt`: PyTorch-native checkpoint
    - `model.torchpp`: Custom format for loading in C++ (Torch++)

**2. Inference (`components/run_inference.py`)**

- Loads the trained model.
- Runs inference `n` times using the full dataset with varying batch sizes `b`.
- Measures execution time and saves results to `data/inference_times.csv`.
    
    Parameters:
        - `n` ∈ {1, 10, 100, 1000, 10000}
        - `b` ∈ {1, 6, 16, 64}

**3. PlotResults (`components/plot.py`)**

- Loads `inference_times.csv`.
- Generates performance visualizations using `matplotlib`.

**4. Tests (`tests/sample.py`)**

- Generates a fixed input-output pair with `batch_size=4`.
- Exports it to `data/example.csv` for cross-platform validation.

C++ Validation 
-----------------------------------------------------------
(`test/basic_net/correct_calculations.cpp`)

- Implements a reference version of the model in pure C++.
- Uses `example.csv` to compare inference results against PyTorch.
- Relies on a custom `Tensor` class for matrix and tensor operations.

Dependencies
------------

Install Python dependencies using:

.. code-block:: bash

    pip install -r requierments.txt

Required packages:

- `torch`
- `scikit-learn`
- `pandas`
- `tqdm`
- `matplotlib`

Usage
-----

Run the full pipeline with:

.. code-block:: bash

    python main.py

This performs:

- Dataset generation and model training
- Inference benchmarking
- Plotting of inference timings
- Export of a sample batch for C++ validation

Features and Optimizations
--------------------------

First and Last Touch Primitives in Einsum Trees
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to use activation functions like ReLU, we have to be able to add markers for first and last touch primitives to each node. 
Thus, we continued our standard for describing the einsum tree by adding multiple characters, which each represent a different first/last touch primitive.
For example 'r' stands for ReLU and 'z' for zero. These characters have to be positioned correctly for them to be used. A first touch primitive has to be 
inside the brackets, of the respective tensor holding the tensor dimension IDs. For example, if we have a tensor with dimension IDs :code:`[0, 1, 3]`, the first 
touch primitive 'z' has to be placed like this:

.. code-block:: text

    [0, 1, 3, z]

If we want to use a last touch primitive, it has to be placed after the brackets, like this: 

.. code-block:: text

    [0, 1, 3]r


Bias Implementation
^^^^^^^^^^^^^^^^^^^

The bias of a neural network layer is a simple tensor that is added to the output of the layer. In our task it is specifically a vector that has to be added to 
each row of each output matrix. Thus, our first simple implementation was to pass the bias vector into our recursive execution function of the einsum tree. An 
extra addition was performed after the execution of a contraction using the correctly marked bias vector. This implementation was simple but not very efficient, 
as it required an additional loop over the output matrix after each contraction. Therefore, we used the idea of a classmate, which was given to us after the 
final presentation. Our current implementation load the respective bias vector into the output tensor of each contraction. This is implemented by filling each 
column with the corresponding bias value as we work on column-major tensors. This way, we can avoid the additional additions and directly load ("add") the bias 
to our (intermediate-)output tensors.

We all worked on the tasks in equal parts.