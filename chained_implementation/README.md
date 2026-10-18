# Chained Implementation — PyTorch–FPGA (Kria KV260)

Interfaz de inferencia para ejecutar **LeNet-5** y **MobileNetV2** en una **AMD Kria KV260**, mediante kernels HLS, XRT y un grafo de ejecución. La interfaz conecta Python con C++ mediante **pybind11** y permite reutilizar buffers XRT entre operaciones, sin devolver obligatoriamente cada resultado intermedio a Python.

## 1. Requisitos

Configuración en la que se ha ejecutado el proyecto:

| Componente | Versión |
|---|---|
| Tarjeta | AMD Kria KV260 |
| Sistema operativo | Ubuntu 22.04.4 LTS (aarch64) |
| Python | 3.10.12 |
| Compilador | g++ 11.4.0 (C++17) |
| XRT | 2.13.479 |
| xmutil | 1.2ubuntu1 |
| pybind11 | 2.9.1 |
| NumPy | 1.21.5 |
| Pillow | 9.0.1 |
| PyTorch | 2.14.0+cpu |
| Torchvision | 0.29.0+cpu |

Se requieren las cabeceras de Python y XRT, además de `uuid-dev` y las bibliotecas `libxrt_coreutil`, `libuuid` y pthread.

Comprobar las herramientas y bibliotecas del sistema:

```bash
python3 --version
g++ --version
dpkg -l | grep -E 'xrt|xmutil|dfx-mgr|python3-dev|uuid-dev'
ls /usr/include/xrt
ldconfig -p | grep -E 'xrt_coreutil|libuuid'
```

> La construcción del bitstream **no** forma parte de este procedimiento: se utiliza la aplicación FPGA ya compilada. Para regenerar el hardware se necesita un flujo de desarrollo HLS/Vitis independiente.

## 2. Preparar el entorno Python

Desde la raíz de `chained_implementation`:

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
```

Instalar las dependencias:

```bash
python3 -m pip install -r requirements.txt
```

Comprobar las versiones instaladas:

```bash
python3 -c "import torch, torchvision, numpy, PIL, pybind11; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('numpy:', numpy.__version__); print('Pillow:', PIL.__version__); print('pybind11:', pybind11.__version__)"
```


## 3. Instalar la aplicación FPGA

La carpeta `fpga_accelerators/` del repositorio debe contener:

```text
fpga_accelerators/
├── fpga_kernels.bit.bin
├── fpga_kernels.dtbo
├── fpga_kernels.xclbin
└── shell.json
```

Instalar el paquete en la Kria:

```bash
sudo mkdir -p /lib/firmware/xilinx/fpga_accelerators
sudo cp fpga_accelerators/* /lib/firmware/xilinx/fpga_accelerators/
```

Comprobar y cargar la aplicación:

```bash
sudo xmutil listapps
sudo xmutil loadapp fpga_accelerators
sudo xmutil listapps
```

Si otra aplicación ocupa el slot, descargarla antes con `sudo xmutil unloadapp` (esto interrumpe el diseño FPGA activo). La aplicación cargada debe aparecer con un `Active_slot` distinto de `-1`.

El runtime busca el XCLBIN en la ruta fija:

```text
/lib/firmware/xilinx/fpga_accelerators/fpga_kernels.xclbin
```

## 4. Compilar la interfaz

Desde la raíz del proyecto, con el entorno virtual activado y el firmware instalado:

```bash
make
```

Se genera el módulo de Python `fpga_dag*.so`, compilado a partir del runtime XRT, los nueve wrappers C++ y el binding pybind11.

Para recompilar:

```bash
make rebuild
```

El `Makefile` configura automáticamente `PYTHONPATH` para sus pruebas.

## 5. Ejecutar pruebas

| Comando | Función |
|---|---|
| `make test-dag` | Probar el binding del DAG |
| `make test-wrappers` | Probar wrappers Python |
| `make test-reshape` | Probar reshape simbólico |
| `make test-lenet5 LIMIT=100` | Evaluar 100 imágenes MNIST |
| `make test-lenet5` | Evaluar 10 000 imágenes MNIST |
| `make test-mobilenetv2 IMAGE=test_images/test_1.jpeg` | Ejecutar MobileNetV2 con una imagen |

### Datos y pesos

- **LeNet-5:** utiliza `weights/lenet5_reference.pth` y el conjunto de prueba de **MNIST**. El script `tests/test_lenet5_fpga.py` admite una opción de descarga de MNIST; consultar `--help` para su sintaxis exacta. El target `make test-lenet5` no activa explícitamente esa opción, por lo que MNIST debe estar disponible antes de ejecutarlo.
- **MobileNetV2:** utiliza pesos preentrenados de torchvision o pesos locales, según los argumentos del script. La primera ejecución puede requerir acceso a Internet para obtener los pesos. Las imágenes de ejemplo están en `test_images/`.

Para consultar directamente las opciones de los scripts, configurar las rutas de importación:

```bash
export PYTHONPATH="$PWD:$PWD/host:$PWD/host/bindings:$PWD/host/python_wrappers${PYTHONPATH:+:$PYTHONPATH}"
python3 tests/test_lenet5_fpga.py --help
python3 tests/test_mobilenetv2_fpga.py --help
```

