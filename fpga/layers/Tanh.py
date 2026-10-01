import numpy as np
import torch
from torch import nn

import fpga_kernels


class FpgaTanh(nn.Module):         # Hereda nn.Module para usarse dentro de un modelo igual a las capas Pythorch, el calculo lo hace FPGA
    def forward(                   # PyTorch llama a este metodo cuando se hace un forward pass del modelo
        self,                      # cuando se hace y = tanh(x), se llama a este metodo con x como input_tensor y termina ejecuntado FpgaTanh.forward(x)
        input_tensor: torch.Tensor
    ) -> torch.Tensor: 

        original_device = input_tensor.device    # recuerda donde se estaba llamando el tensor originalmente en este caso en CPU , para devolver el resultado al mismo dispositivo
 
        input_numpy = np.ascontiguousarray(      # convierte el tensor de entrada a un array de numpy, que es lo que espera la funcion de la FPGA, y lo hace contiguo en memoria para que sea mas eficiente
            input_tensor.detach().cpu().numpy(), # detach() para que no se haga tracking de gradientes, cpu() para moverlo a CPU y numpy() para convertirlo a un array de numpy
            dtype=np.float32
        )

        output_numpy = fpga_kernels.tanh_forward( # ocurre la conexion con la FPGA y se ejecuta el kernel de tanh, que es la funcion de activacion que se quiere calcular
            input_numpy                           # basicamente input_numpy -> pybind11 -> C++ (crea/prepara buffers XRT, copia entrada CPU - FPGA) -> Kernel tanh_forward -> FPGA (copiar resultado FPGA - CPU) -> C++ -> pybind11 -> output_numpy
        )

        output_tensor = torch.from_numpy(         # se hace la conversión inversa de numpy a tensor de PyTorch, para que pueda ser usado en el modelo de PyTorch
            output_numpy
        )
 
        return output_tensor.to(original_device) # regresa al dispositivo original donde estaba el tensor de entrada, para que el modelo pueda seguir funcionando correctamente (CPU)+