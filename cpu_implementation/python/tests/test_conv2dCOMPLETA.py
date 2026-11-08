import torch

from python.layers.Conv2d import ManualConv2d


TOLERANCE = 1e-5


def manual_to_tensor(x):
    return torch.tensor(
        x,
        dtype=torch.float32,
    )


def copy_manual_parameters_to_torch(
    manual_layer,
    torch_layer,
):
    with torch.no_grad():

        torch_weight = torch.tensor(
            manual_layer.weight,
            dtype=torch.float32,
        )

        torch_layer.weight.copy_(
            torch_weight
        )

        if manual_layer.bias is not None:
            torch_bias = torch.tensor(
                manual_layer.bias,
                dtype=torch.float32,
            )

            torch_layer.bias.copy_(
                torch_bias
            )


def calculate_errors(
    manual_output,
    torch_output,
):
    manual_tensor = manual_to_tensor(
        manual_output
    )

    error = (
        manual_tensor
        - torch_output
    )

    rmse = torch.sqrt(
        torch.mean(
            error ** 2
        )
    )

    max_error = torch.max(
        torch.abs(
            error
        )
    )

    return (
        rmse.item(),
        max_error.item(),
    )


def run_test(
    name,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    bias,
    input_shape,
):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    # --------------------------------------------------------
    # Capa manual
    # --------------------------------------------------------

    manual_layer = ManualConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )

    # --------------------------------------------------------
    # Capa de referencia PyTorch
    # --------------------------------------------------------

    torch_layer = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )

    # --------------------------------------------------------
    # Copiar exactamente los mismos pesos
    # --------------------------------------------------------

    copy_manual_parameters_to_torch(
        manual_layer,
        torch_layer,
    )

    # --------------------------------------------------------
    # Entrada común
    # --------------------------------------------------------

    torch.manual_seed(0)

    input_tensor = torch.randn(
        input_shape,
        dtype=torch.float32,
    )

    # ManualConv2d trabaja con listas
    manual_input = input_tensor.tolist()

    # --------------------------------------------------------
    # Ejecución manual
    # --------------------------------------------------------

    manual_output = manual_layer(
        manual_input
    )

    # --------------------------------------------------------
    # Ejecución PyTorch
    # --------------------------------------------------------

    with torch.no_grad():

        torch_output = torch_layer(
            input_tensor
        )

    # --------------------------------------------------------
    # Conversión de la salida manual solo para comparar
    # --------------------------------------------------------

    manual_output_tensor = manual_to_tensor(
        manual_output
    )

    print(
        "Input shape:         ",
        tuple(
            input_tensor.shape
        ),
    )

    print(
        "Manual output shape: ",
        tuple(
            manual_output_tensor.shape
        ),
    )

    print(
        "PyTorch output shape:",
        tuple(
            torch_output.shape
        ),
    )

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    rmse, max_error = calculate_errors(
        manual_output,
        torch_output,
    )

    print(
        f"RMSE:      {rmse:.10e}"
    )

    print(
        f"Max error: {max_error:.10e}"
    )

    if (
        rmse <= TOLERANCE
        and
        max_error <= TOLERANCE
    ):

        print("Resultado: PASS")
        return True

    print("Resultado: FAIL")
    return False


# ============================================================
# RESULTADOS
# ============================================================

results = []


# ============================================================
# TEST 1
# Conv2D normal
# ============================================================

results.append(
    run_test(
        name="Test 1 - Conv2D normal",
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=True,
        input_shape=(
            2,
            3,
            8,
            8,
        ),
    )
)


# ============================================================
# TEST 2
# Pointwise Conv2D
# ============================================================

results.append(
    run_test(
        name="Test 2 - Pointwise Conv2D",
        in_channels=4,
        out_channels=8,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        input_shape=(
            2,
            4,
            8,
            8,
        ),
    )
)


# ============================================================
# TEST 3
# Grouped Conv2D
# ============================================================

results.append(
    run_test(
        name="Test 3 - Grouped Conv2D",
        in_channels=4,
        out_channels=8,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=2,
        bias=True,
        input_shape=(
            2,
            4,
            8,
            8,
        ),
    )
)


# ============================================================
# TEST 4
# Depthwise Conv2D
# ============================================================

results.append(
    run_test(
        name="Test 4 - Depthwise Conv2D",
        in_channels=4,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=4,
        bias=False,
        input_shape=(
            2,
            4,
            8,
            8,
        ),
    )
)


# ============================================================
# TEST 5
# Dilation
# ============================================================

results.append(
    run_test(
        name="Test 5 - Conv2D con dilation",
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=2,
        dilation=2,
        groups=1,
        bias=True,
        input_shape=(
            1,
            3,
            8,
            8,
        ),
    )
)


# ============================================================
# TEST 6
# MobileNetV2 - Depthwise 32 canales
#
# Ejemplo representativo:
#
# [1, 32, 112, 112]
#        ↓
# Depthwise 3x3
# groups = 32
#        ↓
# [1, 32, 112, 112]
# ============================================================

results.append(
    run_test(
        name=(
            "Test 6 - MobileNetV2 "
            "Depthwise 32 canales"
        ),
        in_channels=32,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=32,
        bias=False,
        input_shape=(
            1,
            32,
            112,
            112,
        ),
    )
)


# ============================================================
# TEST 7
# MobileNetV2 - Depthwise con stride 2
#
# Ejemplo:
#
# [1, 96, 56, 56]
#        ↓
# Depthwise 3x3
# stride = 2
# groups = 96
#        ↓
# [1, 96, 28, 28]
# ============================================================

results.append(
    run_test(
        name=(
            "Test 7 - MobileNetV2 "
            "Depthwise stride 2"
        ),
        in_channels=96,
        out_channels=96,
        kernel_size=3,
        stride=2,
        padding=1,
        dilation=1,
        groups=96,
        bias=False,
        input_shape=(
            1,
            96,
            56,
            56,
        ),
    )
)


# ============================================================
# TEST 8
# MobileNetV2 - Pointwise 1x1
#
# Ejemplo:
#
# [1, 32, 56, 56]
#        ↓
# Conv 1x1
# 32 → 96 canales
#        ↓
# [1, 96, 56, 56]
# ============================================================

results.append(
    run_test(
        name=(
            "Test 8 - MobileNetV2 "
            "Pointwise 1x1"
        ),
        in_channels=32,
        out_channels=96,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        input_shape=(
            1,
            32,
            56,
            56,
        ),
    )
)


# ============================================================
# TEST 9
# Mini bloque convolucional tipo Inverted Residual
#
# Se prueban tres operaciones consecutivas:
#
# 32
# ↓
# Pointwise 1x1
# 32 → 96
# ↓
# Depthwise 3x3
# 96 → 96
# ↓
# Pointwise 1x1
# 96 → 24
#
# Para evitar que Python puro tarde demasiado,
# usamos resolución espacial 8x8.
# ============================================================


print()
print("=" * 70)
print("Test 9 - Mini bloque MobileNetV2")
print("=" * 70)


torch.manual_seed(0)

block_input = torch.randn(
    (
        1,
        32,
        8,
        8,
    ),
    dtype=torch.float32,
)


# ------------------------------------------------------------
# Etapa 1
# Pointwise expansion
# 32 -> 96
# ------------------------------------------------------------

manual_pw1 = ManualConv2d(
    in_channels=32,
    out_channels=96,
    kernel_size=1,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=False,
)

torch_pw1 = torch.nn.Conv2d(
    in_channels=32,
    out_channels=96,
    kernel_size=1,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=False,
)

copy_manual_parameters_to_torch(
    manual_pw1,
    torch_pw1,
)


manual_stage1 = manual_pw1(
    block_input.tolist()
)

with torch.no_grad():

    torch_stage1 = torch_pw1(
        block_input
    )


# ------------------------------------------------------------
# Etapa 2
# Depthwise
# 96 -> 96
# ------------------------------------------------------------

manual_dw = ManualConv2d(
    in_channels=96,
    out_channels=96,
    kernel_size=3,
    stride=1,
    padding=1,
    dilation=1,
    groups=96,
    bias=False,
)

torch_dw = torch.nn.Conv2d(
    in_channels=96,
    out_channels=96,
    kernel_size=3,
    stride=1,
    padding=1,
    dilation=1,
    groups=96,
    bias=False,
)

copy_manual_parameters_to_torch(
    manual_dw,
    torch_dw,
)


manual_stage2 = manual_dw(
    manual_stage1
)

with torch.no_grad():

    torch_stage2 = torch_dw(
        torch_stage1
    )


# ------------------------------------------------------------
# Etapa 3
# Pointwise projection
# 96 -> 24
# ------------------------------------------------------------

manual_pw2 = ManualConv2d(
    in_channels=96,
    out_channels=24,
    kernel_size=1,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=False,
)

torch_pw2 = torch.nn.Conv2d(
    in_channels=96,
    out_channels=24,
    kernel_size=1,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=False,
)

copy_manual_parameters_to_torch(
    manual_pw2,
    torch_pw2,
)


manual_stage3 = manual_pw2(
    manual_stage2
)

with torch.no_grad():

    torch_stage3 = torch_pw2(
        torch_stage2
    )


# ------------------------------------------------------------
# Comparación final del bloque
# ------------------------------------------------------------

manual_block_output = manual_to_tensor(
    manual_stage3
)

block_error = (
    manual_block_output
    - torch_stage3
)

block_rmse = torch.sqrt(
    torch.mean(
        block_error ** 2
    )
).item()

block_max_error = torch.max(
    torch.abs(
        block_error
    )
).item()


print(
    "Input shape: ",
    tuple(
        block_input.shape
    ),
)

print(
    "Stage 1 - Pointwise:",
    tuple(
        torch_stage1.shape
    ),
)

print(
    "Stage 2 - Depthwise:",
    tuple(
        torch_stage2.shape
    ),
)

print(
    "Stage 3 - Pointwise:",
    tuple(
        torch_stage3.shape
    ),
)

print(
    f"RMSE:      {block_rmse:.10e}"
)

print(
    f"Max error: {block_max_error:.10e}"
)


if (
    block_rmse <= TOLERANCE
    and
    block_max_error <= TOLERANCE
):

    print("Resultado: PASS")
    results.append(True)

else:

    print("Resultado: FAIL")
    results.append(False)


# ============================================================
# RESULTADO GENERAL
# ============================================================

print()
print("=" * 70)
print("RESULTADO GENERAL")
print("=" * 70)

passed = sum(results)

total = len(results)

print(
    f"Pruebas superadas: {passed}/{total}"
)

if all(results):

    print(
        "TODAS LAS PRUEBAS: PASS"
    )

else:

    print(
        "ALGUNA PRUEBA: FAIL"
    )