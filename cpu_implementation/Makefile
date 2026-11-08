# ************************************************************
# Makefile - Implementacion CPU de kernels para PyTorch
# Plataforma: Xilinx Kria KV260
# ************************************************************

CXX := g++

PYTHON ?= python

# Compilacion C++
CXXFLAGS := -O3 -Wall -Wextra -shared -std=c++17 -fPIC

# Includes de Python y pybind11
PYBIND_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)

# Includes de los kernels CPU
INCLUDES := -Ikernels

# Sufijo correspondiente a la version y arquitectura de Python
EXT_SUFFIX := $(shell $(PYTHON) -c \
	"import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Binding CPU y todos los kernels C++
SOURCES := bindings/pybind_module.cpp \
           $(wildcard kernels/*.cpp)

# Extension Python resultante
TARGET := cpp_kernels$(EXT_SUFFIX)

.PHONY: all clean rebuild check

# Compilar el modulo CPU
all: $(TARGET)

$(TARGET): $(SOURCES) $(wildcard kernels/*.hpp)
	$(CXX) $(CXXFLAGS) \
		$(PYBIND_INCLUDES) \
		$(INCLUDES) \
		$(SOURCES) \
		-o $(TARGET)
	@echo ""
	@echo "****************************************"
	@echo "Compilacion CPU finalizada"
	@echo "Modulo generado: $(TARGET)"
	@echo "****************************************"

# Comprobar que Python puede importar el modulo
check: all
	PYTHONPATH=.:python:python/wrappers $(PYTHON) -c \
		"import cpp_kernels; print('Modulo cpp_kernels importado correctamente')"

# Eliminar unicamente la extension generada
clean:
	rm -f cpp_kernels*.so

# Limpiar y recompilar
rebuild: clean all

# Ejecución de modelos C++ CPU
PYTHON := python
PYTHONPATH_CPU := .:python:python/wrappers

.PHONY: run-lenet5 run-mobilenetv2

# Ejecutar LeNet-5
run-lenet5:
	PYTHONPATH=$(PYTHONPATH_CPU) $(PYTHON) python/apps/run_cpp_lenet5.py

# Ejecutar MobileNetV2 con una imagen predeterminada
MOBILENET_IMAGE := test_images/test_1.jpeg

run-mobilenetv2:
	PYTHONPATH=$(PYTHONPATH_CPU) $(PYTHON) python/apps/run_cpp_mobilenetv2.py $(MOBILENET_IMAGE)