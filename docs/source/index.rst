

.. note::

   *Hello this is our project report :).* 

   This project is part of the lecture **"Machine Learning Compiler Lab"**  
   at **Friedrich Schiller Universität Jena**, taught by *Alexander Breuer* in 2025.  
   Developed by **Stefan Remke**, **Phillip Rothenbeck**, and **Tom Schmidt**.

   📂 The source code is available at:  
   `GitHub Repository <https://github.com/stefan0re/machine_learning_compiler>`_

   This documentation outlines the various stages and tasks completed throughout the course,
   including implementation challenges, optimization strategies, and performance evaluations.


==============================================================
Optimised Tensor Contraction Library (C++ | aarch64 | NEON ASM)
==============================================================

.. image:: https://img.shields.io/badge/Platform-aarch64-green
    :target: https://github.com/stefan0re/machine_learning_compiler/
.. image:: https://img.shields.io/badge/Language-C++17-blue
    :target: https://github.com/stefan0re/machine_learning_compiler/
.. image:: https://img.shields.io/badge/License-MIT-lightgrey

High-performance tensor contraction library with **Just-in-Time (JIT)** kernel compilation for **NEON/Assembly**, optimized for **ARM aarch64** platforms. Designed for **machine learning**, **scientific computing**, and **high-throughput tensor operations**.

⚡ Features
==========

- 🚀 **JIT Kernel Compilation**  
  Runtime generation and compilation of high-performance **NEON/Assembly** kernels.

- 🧠 **GEMM and BRGEMM Kernels**  
  Optimized matrix and batched matrix multiplication operations.

- 🔧 **Unary Kernels**  
  Efficient implementations of:
  
  - `ReLU`
  - `Zero`
  - `Transpose`
  - `Identity`

- 🔬 **Einsum Notation Support**  
  Advanced tensor operations using Einstein summation with support for:

  - Multiple tensors
  - Inline ReLU operations
  - Optimizations for machine learning workflows

- ♻️ **Automatic Optimizations**  
  Multiple levels of auto-optimization for faster and more efficient tensor contractions.

📊 Benchmarks & Examples
=========================

- ✅ Micro-benchmarks for all core kernels
- 🧩 Example NEON and assembly kernel code
- 🤖 Real-world machine learning comparison with **PyTorch**
- 🔁 Full training loop demo with dataset integration and metric tracking

🧪 Testing & Validation
=======================

The project includes robust test coverage:

- 🧪 Unit tests using `Catch2` and `CTest`
- ✅ Continuous integration ready
- 🔍 Thorough validation of all core kernels and operations

🛠️ Requirements
===============

**Hardware:**  
- ARM `aarch64` CPU

**Software:**  
- C++17
- CMake
- OpenMP
- Python ≥ 3.8

**Python Dependencies:**

.. code-block:: bash

    pip install -U sphinx furo torch scikit-learn pandas tqdm matplotlib

📚 Building the Documentation
=============================

.. code-block:: bash

    # Create virtual environment
    python -m venv env_sphinx

    # Activate and install dependencies
    source env_sphinx/bin/activate  # (or .\env_sphinx\Scripts\activate on Windows)
    pip install -U sphinx furo

    # Build HTML docs
    make html

🚀 Getting Started
==================

????? 

📜 License
==========

This project is licensed under the MIT License.


.. toctree::
   :hidden:
   :numbered: 0
   :maxdepth: 2

   self
   chapters/week_01.rst
   chapters/week_02.rst
   chapters/week_03.rst
   chapters/week_04.rst
   chapters/week_05.rst
   chapters/week_06.rst
   chapters/week_07.rst
   chapters/week_08.rst
   chapters/week_09.rst
   chapters/week_10.rst
   chapters/individual_phase.rst
