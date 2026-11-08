#include "AvgPool2d.hpp"

#include <algorithm> //Permite usar std::min y std::max


// Hace basicamente Averge Pooling = seleccionar ventana -> sumar -> dividir -> guardar 

void avg_pool2d_forward(  //Implementa manualmente el forwarde AvgPool2d
    const float* input,   //Puntero a memoria - Entrada
    float* output,        //Salida
    int batch_size,       // N - Dimensiones de entra N x C x H x W
    int input_channels,   // C
    int input_height,     // H
    int input_width,      // W
    int output_height,    //Dimensiones de salida H
    int output_width,     //Dimensiones de salida W
    int kernel_height,    //Parametros del Pooling
    int kernel_width,     
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    bool count_include_pad, //controlan cómo calcular el divisor.
    bool use_divisor_override,
    int divisor_override
)
{
    for (
        int batch_index = 0;
        batch_index < batch_size;
        batch_index++
        /*
         *Este for recorre el batch 
         */

    ) {
        for (
            int channel_index = 0;
            channel_index < input_channels;
            channel_index++
           /*
            *Este for recorre los canales
            */

        ) {
            for (
                int output_row = 0;
                output_row < output_height;
                output_row++

                /*
                 * Este for recorre las filas de la salida
                 */
            ) {
                for (
                    int output_column = 0;
                    output_column < output_width;
                    output_column++
                    /*
                     * Este for recorre las columnas de la salida
                     */

/* O sea hasta acá lo que ocurre es batch -> chanels -> output_row -> output_column.
 * for cada imagen
 *    para cada canal
 *        para cada fila de salida
 *            para cada columna de salida
 *                calcular promedio
 *
 */

                ) {
                    const int window_start_row =
                        output_row * stride_height
                        - padding_height;
                    

                    const int window_start_column =
                        output_column * stride_width
                        - padding_width;

                    /* Esto dertermina en qué posición comienza la ventana de pooling en la imagen de entrada, considerando el stride y el padding.
                     * star = output_row * stride_height - padding_height
                     * star = output_column * stride_width - padding_width
                     */
                    const int padded_end_row =
                        std::min(
                            window_start_row
                                + kernel_height,
                            input_height
                                + padding_height
                        );

                    const int padded_end_column =
                        std::min(
                            window_start_column
                                + kernel_width,
                            input_width
                                + padding_width
                        );
                    /* Esto determina en qué posición termina la ventana de pooling en la imagen de entrada, considerando el tamaño del kernel y el padding.
                     * padded_end_row = window_start_row + kernel_height
                     * padded_end_column = window_start_column + kernel_width
                     */
                    

                    const int valid_start_row =
                        std::max(
                            window_start_row,
                            0
                        );

                    const int valid_start_column =
                        std::max(
                            window_start_column,
                            0
                        );

                    const int valid_end_row =
                        std::min(
                            padded_end_row,
                            input_height
                        );

                    const int valid_end_column =
                        std::min(
                            padded_end_column,
                            input_width
                        );
                        /* valid_stard y valid_end determinan los límites de la ventana de pooling
                         * que están dentro de la imagen de entrada, es decir, los límites válidos.
                         * o sea que si la venta empieza fuera de la imagen debudo al padding, no se
                         * puede acceder a ella input [-1] seria memoria invalida.
                         */



                    const int pool_height =
                        padded_end_row
                        - window_start_row;

                    const int pool_width =
                        padded_end_column
                        - window_start_column;

                    const int pool_size =
                        pool_height * pool_width;

                       /* esto determina el tamaño de la ventana de pooling, considerando el padding.
                        *
                        */

                    float accumulated_value = 0.0f; //aquí comienza la suma, crea el acumulador en O

                    for (
                        int input_row = valid_start_row;
                        input_row < valid_end_row;
                        input_row++
                        // recorre las filas válidas
                    ) {
                        for (
                            int input_column = valid_start_column;
                            input_column < valid_end_column;
                            input_column++
                            //recorre las columnas válidas
                        ) {
                            const int input_position =
                                (
                                    (
                                        batch_index
                                        * input_channels
                                        + channel_index
                                    )
                                    * input_height
                                    + input_row
                                )
                                * input_width
                                + input_column;

                               /*Esto convierte 
                                *batch
                                *channel
                                *row
                                *column
                                *en una sola posición del arreglo lineal, no es realmente una matriz 4D, es memoria contnuar
                                *input:[0][1][2][3][4][5][6][7][8]...
                                *la formula (N,C,H,W) se transforma en un unico indice
                                */
                            accumulated_value += input[  //se hace la suma accumulated_value = accumulated_value + input
                                input_position
                            ];
                        }
                    }

                    const int valid_height =
                        valid_end_row
                        - valid_start_row;

                    const int valid_width =
                        valid_end_column
                        - valid_start_column;

                    const int valid_value_count =
                        valid_height * valid_width;

                        // Esto responde a cuántos reales de la entrada participaron en la suma, es decir, cuántos valores válidos se sumaron.
                     
                        

                    int divisor;

                    if (use_divisor_override) { // Divisor personalizado cuando es True
                        divisor = divisor_override;
                    } else if (count_include_pad) { // Contar padding
                        divisor = pool_size;
                    } else {
                        divisor = valid_value_count; // No contar padding
                    }

                    const float average_value =
                        accumulated_value
                        / static_cast<float>(divisor);

                        /*
                        * averge = sum / divisor
                        */

                    const int output_position =
                        (
                            (
                                batch_index
                                * input_channels
                                + channel_index
                            )
                            * output_height
                            + output_row
                        )
                        * output_width
                        + output_column;

                    output[output_position] = //guarda el resultado
                        average_value;        
                        
                }
            }
        }
    }
}
