class ManualLayerAdd:

    def __init__(self, alpha=1.0):

        if isinstance(alpha, bool):
            raise TypeError(
                "alpha debe ser un número, no un booleano."
            )

        if not isinstance(alpha, (int, float)):
            raise TypeError(
                "alpha debe ser un número."
            )

        self.alpha = float(alpha)

    def forward(self, input1, input2):

        return self._add_recursive(
            input1,
            input2
        )

    def _add_recursive(self, input1, input2):

        input1_is_list = isinstance(
            input1,
            list
        )

        input2_is_list = isinstance(
            input2,
            list
        )

        if input1_is_list and input2_is_list:

            if len(input1) != len(input2):
                raise ValueError(
                    "Las entradas deben tener "
                    "la misma forma."
                )

            output = []

            for i in range(len(input1)):

                value = self._add_recursive(
                    input1[i],
                    input2[i]
                )

                output.append(value)

            return output

        if input1_is_list != input2_is_list:

            raise ValueError(
                "Las entradas deben tener "
                "la misma estructura."
            )

        if isinstance(input1, bool):
            raise TypeError(
                "input1 contiene un booleano."
            )

        if isinstance(input2, bool):
            raise TypeError(
                "input2 contiene un booleano."
            )

        if not isinstance(
            input1,
            (int, float)
        ):
            raise TypeError(
                "input1 contiene un valor "
                "no numérico."
            )

        if not isinstance(
            input2,
            (int, float)
        ):
            raise TypeError(
                "input2 contiene un valor "
                "no numérico."
            )

        return (
            input1
            + self.alpha * input2
        )

    def __call__(
        self,
        input1,
        input2
    ):

        return self.forward(
            input1,
            input2
        )