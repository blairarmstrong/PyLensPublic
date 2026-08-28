Output Transformations
========================
Basic Input Types:
-------------------

DOT_PRODUCT:

This is by far the most common input type. It computes the dot product of the incoming weight vector with the incoming activation vector. In other words, it takes the sum over all incoming links of the product of the link weight and the output of the unit the link is coming from. This is the default unless the network is a BOLTZMANN machine.

[v] DISTANCE:

This computes the squared distance between the weight vector and the activation vector. In other words, it takes the sum over all incoming links of the square of the difference between the link weight and the output of the unit the link is coming from. You probably do not want to use this for a backpropagation network. It should be used along with the KOHONEN output type for Kohonen networks.

PRODUCT:

This takes the product of all incoming weights and the outputs of the units from which they come. This can be used to perform the Pi part of a Sigma-Pi unit, which is actually implemented in Lens using more than one unit. PRODUCT could be used for the gating part of a gated unit, which is essentially a Pi-Sigma. Often the weights involved will actually be frozen at 1.0 so that only the sending unit activations are relevant.

[X] IN_BOLTZ:

This is the input half of a Boltzmann unit. If the unit is clamped by either an externalInput or a target within the grace period (see OUT_BOLTZ), this does nothing. Otherwise, it computes a dot product. In the backward pass, this does not propagate derivatives to the sending units but it increments the incoming link derivatives by:
U->output * V->output - U->clampedOutput * V->clampedOutput,
where U is the receiving unit, V is the sending unit, output is the unit's output after the non-clamped phase and clampedOutput is the desired output. This implements the Boltzmann machine learning rule.

IN_COPY:

The units in a group with an IN_COPY input function simply copy their inputs from some field in the corresponding units of another group. The copyConnect command must be used to specify which group and which field will be the source of the copying.

Input Modifying Types:
-----------------------
SOFT_CLAMP:

The SOFT_CLAMP function assumes that the output function is logistic. It adds a factor to the input of the unit such that, with no other input, the output of the unit would be:
initOutput + clampStrength * (externalInput - initOutput)
Thus, the output would fall between the initOutput and the externalInput. The clampStrength, which ranges from 0.0 to 1.0, determines the extent to which the output will be dominated by the externalInput. This is meant to be used by groups that also receive ordinary inputs. The clampStrength should be less than 1.0 if the externalInputs are 0.0 or 1.0 or the group will have infinite input.

[v] INCR_CLAMP:

This function simply adds the externalInput, scaled by the clampStrength, to the unit's input. It is used in interactive activation models, among other things.

IN_INTEGR:

This time-averages the group's input according to the function:
input = lastInput + dt (newInput - lastInput)

It is ordinarily used with CONTINUOUS networks. With a LOGISTIC output function, it differs from OUT_INTEGR in that units will adapt more rapidly when being pulled toward the extremes and less rapidly when being pulled towards an output of 0.5.

[v] IN_NORM:

This normalizes the inputs to the group so they sum to 1.0. This should probably not be used if inputs can be negative because the results may be rather strange.

[v] IN_NOISE:

This makes the input noisy. The noise function is the group's noiseProc and the standard deviation or range is given by the group's noiseRange.

[-] IN_DERIV_NOISE:

This makes the inputDeriv noisy on the backward pass. The noise function is the group's noiseProc and the standard deviation or range is given by the group's noiseRange.
