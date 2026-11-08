CXX := g++

CXXFLAGS := -O2 -shared -std=c++17 -fPIC
PYBIND_INCLUDES := $(shell python3 -m pybind11 --includes)

# Directorios principales
HOST_DIR := host
WRAPPERS_DIR := $(HOST_DIR)/wrappers
BINDINGS_DIR := $(HOST_DIR)/bindings
RUNTIME_DIR := $(HOST_DIR)/runtime

# Includes
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

# Fuentes C++
SOURCES := \
	$(BINDINGS_DIR)/pybind_fpga_module.cpp \
	$(WRAPPERS_DIR)/Tanh/tanh_fpga.cpp \
	$(WRAPPERS_DIR)/Linear/linear_fpga.cpp \
	$(WRAPPERS_DIR)/Conv2d/conv2d_fpga.cpp \
	$(WRAPPERS_DIR)/AvgPool2d/avgpool2d_fpga.cpp \
	$(WRAPPERS_DIR)/Depthwise/depthwise_fpga.cpp \
	$(WRAPPERS_DIR)/Pointwise/pointwise_fpga.cpp \
	$(WRAPPERS_DIR)/BatchNorm2d/batchnorm2d_fpga.cpp \
	$(WRAPPERS_DIR)/ReLU6/relu6_fpga.cpp \
	$(WRAPPERS_DIR)/LayerAdd/layeraad_fpga.cpp \
	$(RUNTIME_DIR)/fpga_runtime.cpp

# Cabeceras
HEADERS := \
	$(WRAPPERS_DIR)/Tanh/tanh_fpga.hpp \
	$(WRAPPERS_DIR)/Linear/linear_fpga.hpp \
	$(WRAPPERS_DIR)/Conv2d/conv2d_fpga.hpp \
	$(WRAPPERS_DIR)/AvgPool2d/avgpool2d_fpga.hpp \
	$(WRAPPERS_DIR)/Depthwise/depthwise_fpga.hpp \
	$(WRAPPERS_DIR)/Pointwise/pointwise_fpga.hpp \
	$(WRAPPERS_DIR)/BatchNorm2d/batchnorm2d_fpga.hpp \
	$(WRAPPERS_DIR)/ReLU6/relu6_fpga.hpp \
	$(WRAPPERS_DIR)/LayerAdd/layeradd_fpga.hpp \
	$(RUNTIME_DIR)/fpga_runtime.hpp

# Módulo Python
TARGET := $(BINDINGS_DIR)/fpga_kernels$(shell python3-config --extension-suffix)

LIBS := -lxrt_coreutil -pthread -luuid

# Compilación
.PHONY: all clean rebuild test-lenet5 test-mobilenetv2

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) \
		$(PYBIND_INCLUDES) \
		$(INCLUDES) \
		$(SOURCES) \
		-o $(TARGET) \
		$(LIBS)


# Limpieza
clean:
	rm -f $(BINDINGS_DIR)/fpga_kernels*.so

# Reconstrucción completa
rebuild: clean all

# Prueba de LeNet-5
test-lenet5: $(TARGET)
	PYTHONPATH="$(CURDIR):$(CURDIR)/$(HOST_DIR):$(CURDIR)/$(BINDINGS_DIR)$${PYTHONPATH:+:$$PYTHONPATH}" \
	python3 -u tests/test_lenet5_fpga.py

# Prueba de MobileNetV2
test-mobilenetv2: $(TARGET)
	@if [ -z "$(IMAGE)" ]; then \
		echo "Uso: make test-mobilenetv2 IMAGE=test_images/test_1.jpeg"; \
		exit 1; \
	fi
	PYTHONPATH="$(CURDIR):$(CURDIR)/$(HOST_DIR):$(CURDIR)/$(BINDINGS_DIR)$${PYTHONPATH:+:$$PYTHONPATH}" \
	python3 -u tests/test_mobilenetv2_fpga.py "$(IMAGE)"