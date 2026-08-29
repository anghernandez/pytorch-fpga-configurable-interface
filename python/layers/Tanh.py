import math


class ManualTanh:

    def __init__(self):
        pass


    def _validate_values(self, x):

        if isinstance(x, list):

            if len(x) == 0:
                raise ValueError(
                    "The input and its dimensions cannot be empty"
                )

            for element in x:
                self._validate_values(element)

        else:

            if (
                not isinstance(x, (int, float))
                or isinstance(x, bool)
            ):
                raise TypeError(
                    "All elements must be numeric"
                )


    def _tanh_scalar(self, value):

        # Se utilizan dos expresiones equivalentes para evitar
        # desbordamientos con valores positivos o negativos grandes.

        if value >= 0.0:

            exponential = math.exp(
                -2.0 * value
            )

            return (
                (1.0 - exponential)
                / (1.0 + exponential)
            )

        exponential = math.exp(
            2.0 * value
        )

        return (
            (exponential - 1.0)
            / (exponential + 1.0)
        )


    def _apply_tanh(self, x):

        if isinstance(x, list):

            output = []

            for element in x:

                output.append(
                    self._apply_tanh(element)
                )

            return output

        return self._tanh_scalar(x)


    def forward(self, x):

        if not isinstance(x, list):
            raise TypeError(
                "The input must be a list or nested list"
            )

        self._validate_values(x)

        return self._apply_tanh(x)


    def __call__(self, x):

        return self.forward(x)


    def __repr__(self):

        return "ManualTanh()"