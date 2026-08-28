from .basic import Basic
from ..array_factory import Array_factory as af

class BoltzmannOutput(Basic):
    func_deriv = None

    def __init__(self, group:"Group"):
        """
        Instantiates the Boltzmann Output transform for passing data fwd in
        Deterministic Boltzmann Machines

        :param group: Group whose output is to be transformed according to BoltzmannOutput
        :type group: Group 
        """
        super().__init__("BoltzmannOutput", group)
        self._update_gain()
        self.dt = self.group.network.dt * self.group.dt # Individual unit dt has not been implemented
        self.unit_data = af.empty(self.group.num_units)

    def _update_gain(self):
        """
        Updates groups gain each foward pass.
        """
        self.gain = (
            self.group.network.gain
            if af.isnan(self.group.gain).any()
            else self.group.gain
        )
        self.gain = af.array(self.gain)

    def _func(self, x):
        """
        Sigmoid function.
        """
        return 1 / (1 + af.exp(-x * self.gain))
            
    def forward(self, x) -> af:
        """
        Passes data forward into the BoltzmannOutput transform

        :param x: the values to pass in
        :type x: ndarray
        """
        self._update_gain()
        output = self.group.output_matrix

        # Store previous group outputs to check if BM has settled
        self.unit_data[:] = output

        for i in range(self.group.num_units):
            if not af.isnan(self.group.external_input[i]):
                output[i] = self.group.external_input[i]

            elif not af.isnan(self.group.target[i]) and self.group.network.in_grace_period:
                output[i] = self.group.target[i]

            else:
                output[i] += self.dt * (
                    self._func(self.group.input_matrix[i]) - output[i]
                )

        return output

