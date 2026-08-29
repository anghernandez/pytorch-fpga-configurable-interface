# PyTorch FPGA Configurable Interface

This project explores the integration of native C++ neural-network layers with PyTorch using **pybind11**, with the goal of validating a C++ implementation against a reference PyTorch model and establishing a software baseline for future FPGA acceleration using **Vitis HLS**.

The current implementation is based on **LeNet-5** and includes native C++ implementations of the main neural-network operations:

* Convolution (`Conv2d`)
* Linear / Fully Connected (`Linear`)
* Hyperbolic Tangent (`Tanh`)
* Average Pooling (`AvgPool2d`)

The C++ kernels are exposed to Python through `pybind11` and wrapped as PyTorch-compatible modules.

The project currently focuses on **forward inference**. PyTorch is used as the reference implementation, while the C++ implementation is validated by comparing predictions and numerical outputs.

---

## Project Structure

```text
.
├── cpp/
│   ├── bindings/
│   │   └── pybind_module.cpp
│   │
│   ├── kernels/
│   │   ├── AvgPool2d.cpp
│   │   ├── Conv2d.cpp
│   │   ├── Linear.cpp
│   │   ├── ReLU.cpp
│   │   └── Tanh.cpp
│   │
│   └── tests/
│       ├── test_avg_pool2d.cpp
│       ├── test_conv2d.cpp
│       ├── test_linear.cpp
│       ├── test_relu.cpp
│       └── test_tanh.cpp
│
├── python/
│   ├── cpp_layers/
│   │   ├── __init__.py
│   │   ├── AvgPool2d.py
│   │   ├── Conv2d.py
│   │   ├── Linear.py
│   │   ├── ReLU.py
│   │   └── Tanh.py
│   │
│   ├── models/
│   │   └── CppLeNet5.py
│   │
│   └── tests/
│       └── test_cpp_lenet5.py
│
├── build/
│   └── cpp_kernels*.so
│
├── requirements.txt
└── README.md
```

---

## Architecture

The software stack is organized into three main layers.

```text
                    PyTorch
                       │
                       ▼
                CppLeNet5
                       │
             Python layer wrappers
                       │
                       ▼
                   pybind11
                       │
                       ▼
                 C++ kernels
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
      Conv2d         Linear          Tanh
        │
        └──────────────┬──────────────┘
                       ▼
                  AvgPool2d
```

The C++ implementations perform the actual forward computations, while the Python wrappers provide a PyTorch-compatible interface.

---

## LeNet-5 Implementation

The current `CppLeNet5` model explicitly defines each operation:

```text
Input
  │
  ▼
Conv2d
  │
  ▼
Tanh
  │
  ▼
AvgPool2d
  │
  ▼
Conv2d
  │
  ▼
Tanh
  │
  ▼
AvgPool2d
  │
  ▼
Flatten
  │
  ▼
Linear
  │
  ▼
Tanh
  │
  ▼
Linear
  │
  ▼
Tanh
  │
  ▼
Linear
  │
  ▼
Output
```

The explicit layer representation is intentional. It allows each operation to be independently tested, profiled, optimized, and eventually ported to hardware.

---

## Weight Loading

The PyTorch model is used as the reference model and provides the trained parameters.

The C++ implementation does not rely on identical PyTorch module names. Instead, `CppLeNet5` explicitly maps the parameters from the PyTorch `state_dict` to the corresponding C++ layers.

For example:

```python
self.conv1.weight.copy_(
    state_dict["features.0.weight"]
)

self.conv2.weight.copy_(
    state_dict["features.3.weight"]
)

self.fc1.weight.copy_(
    state_dict["classifier.1.weight"]
)
```

This makes the relationship between the reference PyTorch architecture and the native C++ implementation explicit.

---

## Requirements

The project requires:

* Python 3.14
* C++ compiler (`g++`)
* GCC
* pybind11
* NumPy
* PyTorch
* torchvision

Create and activate the virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Build

The C++ kernels and `pybind11` bindings must be compiled into a Python extension module.

The resulting shared library is placed in the `build/` directory and imported by the Python layer wrappers.

The exact compilation command depends on the local Python and pybind11 installation.

For example, the pybind11 include directory can be obtained with:

```bash
python -m pybind11 --includes
```

and the Python extension suffix with:

```bash
python3-config --extension-suffix
```

---

## Testing Individual C++ Kernels

The native C++ kernels can be tested independently before integrating them into the complete model.

The tests are located in:

```text
cpp/tests/
```

The main kernels currently covered are:

```text
Conv2d
Linear
Tanh
AvgPool2d
ReLU
```

This separation allows numerical correctness to be established at the kernel level before testing the complete LeNet-5 model.

---

## Testing C++ LeNet-5 Against PyTorch

The complete model can be validated with:

```bash
python python/tests/test_cpp_lenet5.py
```

The test:

1. Loads the trained PyTorch weights.
2. Creates the reference PyTorch LeNet-5 model.
3. Creates the C++-backed `CppLeNet5` model.
4. Loads the same trained parameters into both models.
5. Runs inference on the test dataset.
6. Compares the predictions.
7. Compares the numerical outputs.

The objective is to demonstrate that the native C++ implementation reproduces the PyTorch reference implementation with negligible numerical error.

---

## Validation

The current validation baseline was obtained using 100 test images.

The results were:

```text
Images tested:             100
Same predictions:          100/100
PyTorch accuracy:          99.00%
C++ accuracy:              99.00%
Average RMSE:              8.91773021e-07
Maximum RMSE:              1.86699122e-06
Maximum absolute error:    5.722...
```

These results indicate that the C++ implementation reproduces the PyTorch inference behavior with very small numerical differences.

This validation serves as the software baseline before hardware acceleration.

---

## Development Branch

The Python/C++ wrapper development is currently being performed on:

```text
feature/python-cpp-wrappers
```

The branch contains the explicit PyTorch-compatible wrappers and the C++ implementation used for the LeNet-5 validation.

---

## FPGA / HLS Roadmap

The current C++ implementation is intended to serve as the software reference for the next stage of the project.

The planned acceleration flow is:

```text
PyTorch
   │
   ▼
CppLeNet5
   │
   ▼
C++ kernels
   │
   ▼
HLS-compatible C++
   │
   ▼
Vitis HLS
   │
   ├── C Simulation
   ├── C Synthesis
   └── Co-Simulation
   │
   ▼
RTL / IP
   │
   ▼
Vitis
   │
   ▼
FPGA / Kria
```

The main objective is to progressively port the computational kernels from native C++ to **Vitis HLS**, where C/C++ code is synthesized into hardware.

The explicit layer structure makes it possible to treat each operation as an independent hardware kernel and evaluate its:

* Latency
* Initiation interval
* Resource utilization
* Memory access behavior
* Throughput
* Numerical accuracy

---

## Current Status

### Software

* [x] Native C++ neural-network kernels
* [x] pybind11 integration
* [x] Python PyTorch-compatible wrappers
* [x] C++ LeNet-5 implementation
* [x] Explicit layer representation
* [x] PyTorch weight loading
* [x] Kernel-level testing
* [x] PyTorch vs C++ validation
* [x] Numerical accuracy validation

### FPGA / HLS

* [ ] Adapt C++ kernels for HLS synthesis
* [ ] Create HLS test benches
* [ ] Run C simulation
* [ ] Run C synthesis
* [ ] Analyze latency and resource utilization
* [ ] Run co-simulation
* [ ] Generate hardware IP
* [ ] Integrate IP using Vitis
* [ ] Deploy on FPGA
* [ ] Validate FPGA inference against the PyTorch reference

---

## Objective

The overall objective is to establish a reproducible path from a PyTorch neural-network model to a hardware-accelerated FPGA implementation:

```text
PyTorch
   ↓
Python / C++ integration
   ↓
Validated C++ kernels
   ↓
Vitis HLS
   ↓
Hardware IP
   ↓
FPGA acceleration
```

The PyTorch implementation remains the numerical reference throughout the development process, allowing each subsequent implementation stage to be validated against the same baseline.

```
```
