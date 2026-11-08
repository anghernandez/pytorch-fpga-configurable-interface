import math


class ManualSoftmax:

    def __init__(self, dim=-1):

        if not isinstance(dim, int):
            raise TypeError(
                "dim must be an integer"
            )

        self.dim = dim


    def _get_shape(self, x):

        if not isinstance(x, list):
            return []

        if len(x) == 0:
            raise ValueError(
                "The input and its dimensions cannot be empty"
            )

        first_shape = self._get_shape(x[0])

        for element in x[1:]:

            current_shape = self._get_shape(element)

            if current_shape != first_shape:
                raise ValueError(
                    "The input must be a rectangular nested list"
                )

        return [len(x)] + first_shape


    def _validate_values(self, x):

        if isinstance(x, list):

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


    def _create_output(self, shape):

        if len(shape) == 1:

            output = []

            for _ in range(shape[0]):
                output.append(0.0)

            return output

        output = []

        for _ in range(shape[0]):

            output.append(
                self._create_output(shape[1:])
            )

        return output


    def _read_value(self, x, indices):

        value = x

        for index in indices:
            value = value[index]

        return value


    def _write_value(self, x, indices, value):

        destination = x

        for i in range(len(indices) - 1):
            destination = destination[indices[i]]

        destination[indices[-1]] = value


    def _generate_indices_excluding_dim(
        self,
        shape,
        dim,
        depth=0,
        current_indices=None
    ):

        if current_indices is None:
            current_indices = []

        if depth == len(shape):
            return [current_indices]

        if depth == dim:

            return self._generate_indices_excluding_dim(
                shape,
                dim,
                depth + 1,
                current_indices + [None]
            )

        results = []

        for index in range(shape[depth]):

            new_indices = (
                current_indices + [index]
            )

            generated_indices = (
                self._generate_indices_excluding_dim(
                    shape,
                    dim,
                    depth + 1,
                    new_indices
                )
            )

            for indices in generated_indices:
                results.append(indices)

        return results


    def _softmax_vector(self, vector):

        maximum = vector[0]

        for value in vector:

            if value > maximum:
                maximum = value

        exponentials = []
        sum_exponentials = 0.0

        for value in vector:

            exponential = math.exp(
                value - maximum
            )

            exponentials.append(
                exponential
            )

            sum_exponentials += exponential

        result = []

        for exponential in exponentials:

            result.append(
                exponential / sum_exponentials
            )

        return result


    def forward(self, x):

        if not isinstance(x, list):
            raise TypeError(
                "The input must be a list or nested list"
            )

        shape = self._get_shape(x)

        self._validate_values(x)

        num_dimensions = len(shape)

        dim = self.dim

        if dim < 0:
            dim = num_dimensions + dim

        if dim < 0 or dim >= num_dimensions:
            raise ValueError(
                f"dim={self.dim} is out of range for "
                f"an input with {num_dimensions} dimensions"
            )

        output = self._create_output(shape)

        index_groups = (
            self._generate_indices_excluding_dim(
                shape,
                dim
            )
        )

        for base_indices in index_groups:

            vector = []

            for dim_index in range(shape[dim]):

                indices = base_indices.copy()
                indices[dim] = dim_index

                value = self._read_value(
                    x,
                    indices
                )

                vector.append(value)

            softmax_vector = self._softmax_vector(
                vector
            )

            for dim_index in range(shape[dim]):

                indices = base_indices.copy()
                indices[dim] = dim_index

                self._write_value(
                    output,
                    indices,
                    softmax_vector[dim_index]
                )

        return output


    def __call__(self, x):

        return self.forward(x)


    def __repr__(self):

        return f"ManualSoftmax(dim={self.dim})"