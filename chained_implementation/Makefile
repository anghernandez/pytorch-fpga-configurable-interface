# ------------------------------------------------------------
# CONFIGURACIÓN DEL COMPILADOR
# ------------------------------------------------------------

CXX := g++

CXXFLAGS := -O2 -shared -std=c++17 -fPIC

PYBIND_INCLUDES := $(shell python3 -m pybind11 --includes)

PYTHON := python3

PYTHON_EXTENSION := $(shell python3-config --extension-suffix)


# ------------------------------------------------------------
# DIRECTORIOS DEL PROYECTO
# ------------------------------------------------------------

HOST_DIR := host

WRAPPERS_DIR := $(HOST_DIR)/wrappers

PYTHON_WRAPPERS_DIR := $(HOST_DIR)/python_wrappers

BINDINGS_DIR := $(HOST_DIR)/bindings

RUNTIME_DIR := $(HOST_DIR)/runtime

TESTS_DIR := tests


# ------------------------------------------------------------
# DIRECTORIOS DE INCLUSIÓN C++
# ------------------------------------------------------------

INCLUDES := \
	-I/usr/include/xrt \
	-I/usr/include/xrt/deprecated \
	-I$(WRAPPERS_DIR)/Tanh \
	-I$(WRAPPERS_DIR)/Linear \
	-I$(WRAPPERS_DIR)/Conv2d \
	-I$(WRAPPERS_DIR)/AvgPool2d \
	-I$(WRAPPERS_DIR)/Depthwise \
	-I$(WRAPPERS_DIR)/Pointwise \
	-I$(WRAPPERS_DIR)/BatchNorm2d \
	-I$(WRAPPERS_DIR)/ReLU6 \
	-I$(WRAPPERS_DIR)/LayerAdd \
	-I$(RUNTIME_DIR)


# ------------------------------------------------------------
# FUENTES C++ DE LOS WRAPPERS FPGA
# ------------------------------------------------------------

WRAPPER_SOURCES := \
	$(WRAPPERS_DIR)/Tanh/tanh_fpga.cpp \
	$(WRAPPERS_DIR)/Linear/linear_fpga.cpp \
	$(WRAPPERS_DIR)/Conv2d/conv2d_fpga.cpp \
	$(WRAPPERS_DIR)/AvgPool2d/avgpool2d_fpga.cpp \
	$(WRAPPERS_DIR)/Depthwise/depthwise_fpga.cpp \
	$(WRAPPERS_DIR)/Pointwise/pointwise_fpga.cpp \
	$(WRAPPERS_DIR)/BatchNorm2d/batchnorm2d_fpga.cpp \
	$(WRAPPERS_DIR)/ReLU6/relu6_fpga.cpp \
	$(WRAPPERS_DIR)/LayerAdd/layeradd_fpga.cpp


# ------------------------------------------------------------
# CABECERAS C++ DE LOS WRAPPERS FPGA
# ------------------------------------------------------------

WRAPPER_HEADERS := \
	$(WRAPPERS_DIR)/Tanh/tanh_fpga.hpp \
	$(WRAPPERS_DIR)/Linear/linear_fpga.hpp \
	$(WRAPPERS_DIR)/Conv2d/conv2d_fpga.hpp \
	$(WRAPPERS_DIR)/AvgPool2d/avgpool2d_fpga.hpp \
	$(WRAPPERS_DIR)/Depthwise/depthwise_fpga.hpp \
	$(WRAPPERS_DIR)/Pointwise/pointwise_fpga.hpp \
	$(WRAPPERS_DIR)/BatchNorm2d/batchnorm2d_fpga.hpp \
	$(WRAPPERS_DIR)/ReLU6/relu6_fpga.hpp \
	$(WRAPPERS_DIR)/LayerAdd/layeradd_fpga.hpp


# ------------------------------------------------------------
# RUNTIME XRT
# ------------------------------------------------------------

RUNTIME_SOURCE := $(RUNTIME_DIR)/fpga_runtime.cpp

RUNTIME_HEADER := $(RUNTIME_DIR)/fpga_runtime.hpp


# ------------------------------------------------------------
# BINDINGS PYBIND11 PARA EL DAG
# ------------------------------------------------------------

BINDING_SOURCE := $(BINDINGS_DIR)/pybind_fpga_dag.cpp


# ------------------------------------------------------------
# FUENTES Y DEPENDENCIAS
# ------------------------------------------------------------

SOURCES := \
	$(BINDING_SOURCE) \
	$(WRAPPER_SOURCES) \
	$(RUNTIME_SOURCE)

HEADERS := \
	$(WRAPPER_HEADERS) \
	$(RUNTIME_HEADER)


# ------------------------------------------------------------
# MÓDULO PYTHON GENERADO
# ------------------------------------------------------------

TARGET := fpga_dag$(PYTHON_EXTENSION)

LIBS := -lxrt_coreutil -pthread -luuid


# ------------------------------------------------------------
# PYTHONPATH
# ------------------------------------------------------------

PROJECT_PYTHONPATH := \
$(CURDIR):$(CURDIR)/$(HOST_DIR):$(CURDIR)/$(BINDINGS_DIR):$(CURDIR)/$(PYTHON_WRAPPERS_DIR)

export PYTHONPATH := $(PROJECT_PYTHONPATH):$(PYTHONPATH)


# ------------------------------------------------------------
# COMANDOS DISPONIBLES
# ------------------------------------------------------------

.PHONY: \
	all \
	clean \
	rebuild \
	test-dag \
	test-wrappers \
	test-linear \
	test-reshape \
	test-lenet5 \
	test-mobilenetv2


# ------------------------------------------------------------
# COMPILACIÓN DEL MÓDULO FPGA DAG
# ------------------------------------------------------------

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) \
		$(PYBIND_INCLUDES) \
		$(INCLUDES) \
		$(SOURCES) \
		-o $(TARGET) \
		$(LIBS)


# ------------------------------------------------------------
# LIMPIEZA
# ------------------------------------------------------------

clean:
	rm -f fpga_dag*.so


# ------------------------------------------------------------
# RECONSTRUCCIÓN COMPLETA
# ------------------------------------------------------------

rebuild: clean all


# ------------------------------------------------------------
# PRUEBA DEL BINDING PYTHON - DAG
# ------------------------------------------------------------

test-dag: $(TARGET)
	$(PYTHON) -u $(BINDINGS_DIR)/test_pybind_dag.py


# ------------------------------------------------------------
# PRUEBA DE WRAPPERS PYTHON CON DAG
# ------------------------------------------------------------

test-wrappers: $(TARGET)
	$(PYTHON) -u $(TESTS_DIR)/test_wrappers_dag.py




# ------------------------------------------------------------
# PRUEBA DEL RESHAPE SIMBÓLICO
# ------------------------------------------------------------

test-reshape:
	$(PYTHON) -u $(TESTS_DIR)/test_reshape_dag.py


# ------------------------------------------------------------
# PRUEBA DE LENET-5 CON DAG
# ------------------------------------------------------------

test-lenet5: $(TARGET)
	$(PYTHON) -u $(TESTS_DIR)/test_lenet5_fpga.py \
		--limit $(if $(LIMIT),$(LIMIT),10000)


# ------------------------------------------------------------
# PRUEBA DE MOBILENETV2
# ------------------------------------------------------------

test-mobilenetv2: $(TARGET)
	@if [ -z "$(IMAGE)" ]; then \
		echo "Uso: make test-mobilenetv2 IMAGE=test_images/test_1.jpeg"; \
		exit 1; \
	fi
	$(PYTHON) -u $(TESTS_DIR)/test_mobilenetv2_fpga.py "$(IMAGE)"