Output Transformations
======================

Basic Output Types:
-------------------

LINEAR:
^^^^^^^^

This simply copies the input to the output.

LOGISTIC:
^^^^^^^^^^

Computes the traditional sigmoid function:
O = 1 / (1 + exp(-i * gain))
Gain is the inverse of the temperature. It is used to avoid division. Ordinarily the gain is taken from the network's gain field or the group's gain field if that is set. If ADAPTIVE_GAIN is used, each unit will have its own trainable gain.

[v] TERNARY:

This is essentially a normal sigmoid shifted to the right added to a negated sigmoid of -i shifted to the left. Alternately, you can think of it as a [-1,1] sigmoid that has a flat place at 0. It is designed to give the unit stable outputs at -1, 1, and 0. You could think of such units as coding whether a feature is present, absent, or unknown. The gain affects the slope of each of the two sigmoids. The ternaryShift sets the distance between their centers. Increasing the ternaryShift will make the central plateau wider. Increasing the gain will make the transitions between plateaus sharper.

TANH:

This is equivalent to 1 - 2S(2 i), where S is the ordinary sigmoid function and I is the input. Note that its slope is actually twice what the slope would be if you just stretched a sigmoid to the range [-1,1]. So you may want to use half the normal gain to compensate. If ADAPTIVE_GAIN is used, each unit will have its own trainable gain.

EXPONENTIAL:

This is just exp(i). There is a big potential for overflow with this, so you may want to be careful how you use it.

GAUSSIAN:

This computes a gaussian radial basis function: exp(-i^2 * gain^2). This is often as effective as LOGISTIC, although it can become a bit unstable at the end of training. It can also be used in conjunction with ADAPTIVE_GAIN for individual, trainable gains for each unit.

SOFT_MAX:

This is equivalent to an exponential followed by a normalization. However, SOFT_MAX scales the values before computing the exponential. This doesn't affect the end result but it avoids overflow. A SOFT_MAX OUTPUT group will get DIVERGENCE error by default.

[X] KOHONEN:

This is used for the map layer in a KOHONEN network. It should be combined with a DISTANCE input function. It finds the unit whose weight vector is most similar to the input vector. Any unit in the map whose squared Euclidean distance from the best unit is greater than neighborhood will be silent. Groups in the neighborhood will have output equal to 1.0 minus the ratio between the unit's input and the largest input of any unit. The output will therefore fall in the range [0.0, 1.0].
In the backward pass, the inputDeriv of units in the neighborhood will be set to 1.0 and that of the others to 0.0. Only units in the neighborhood will be able to alter their incoming weights. The DISTANCE procedure, in the backward pass, will cause the incoming weights to drift towards the input vector.

[X] OUT_BOLTZ:

This is used for groups in a Boltzmann network. If the unit has an externalInput, the output will be clamped to that value. Otherwise, if it has a target and the network is in the gracePeriod (the positive phase of the Boltzmann algorithm), the output will be clamped to the target. Otherwise, it is computed as a time-averaged sigmoid of the input.

[v] OUT_COPY: (Displays nan for )

The units in a group with an OUT_COPY output function simply copy their outputs from some field in the corresponding units of another group. The copyConnect command must be used to specify which group and which field will be the source of the copying.

[v] INTERACT_INTEGR:

This implements the interactive-activation output rule. For a traditional interactive-activation model, it should be used with DOT_PRODUCT inputs, and an INCR_CLAMP input function. It contains a decay term and time-averages the activations. The decay is fixed at 1.0. This version crops the unit outputs to the range [0, maxOutput] because negative outputs are not normally used in an IA model.

Clamping Output Types:
-----------------------

HARD_CLAMP:

If the externalInput is a real number, this sets the output to the externalInput. Otherwise it does nothing.

[v] BIAS_CLAMP:

This sets the output to the initOutput (defaults to 1.0 for BIAS groups).

ELMAN_CLAMP:

In order for an ELMAN_CLAMP function to work, you must first use elmanConnect to associate a source group with the context group. This simply copies the (cached) output from each source unit and adds it to the output of the corresponding context unit. It is possible to have more that one ELMAN_CLAMP function. In this case, the output will simply sum the outputs from each of the source units. If a group has multiple ELMAN_CLAMP functions, each call to elmanConnect will define the source group for the first function that has not yet been assigned a group.

[v] WEAK_CLAMP:

This shifts the output a certain fraction of the way towards the externalInput. The fraction is determined by the clampStrength. Specifically, the function is: o = o + clampStrength * (externalInput - o)

Output Modifying Types:
------------------------

OUT_INTEGR:

This is just like IN_INTEGR but it integrates the output rather than the input. This is put on by default in a CONTINUOUS network unless IN_INTEGR is specified.

[v] OUT_NORM:

This normalizes the outputs of the units in the group to sum to 1.0. It probably should not be used unless the un-normalized values are constrained to be positive. The SOFT_MAX function should be used rather than an EXPONENTIAL followed by OUT_NORM because SOFT_MAX will avoid numerical overflow.

OUT_NOISE:

This makes the output noisy. The type of noise is determined by the group's noiseProc and noiseRange parameters.

[-] OUT_DERIV_NOISE: (how to test noise)

This injects noise into the outputDerivs on the backward pass. The type of noise is determined by the group's noiseProc and noiseRange parameters.

OUT_CROPPED:

This crops the output to within the range [minOutput, maxOutput]. You may want to use this after OUT_NOISE to prevent outputs outside of this range.

[v] OUT_WINNER:

This is a winner-take-all filter. The most active unit retains its activation and the other units are set to the minimum output value for the group. In the backward phase, the original outputs are restored to enable error to be backpropagated across the transfer function.

3/9:
Weak Clamp, should it update? see comments

Kohonen: what is periodic boundary, neighbor? do i implement that?

Out_deriv Noise: no forward function, how to ignore

Finished:

Ternary
Out_copy (but displays nan for error rather than 200000)
Bias Clamp
Out_Norm
Out Winner
INTERACT_INTEGR (same as weak clamp)
WEAK_CLAMP (done but theres no actual change, just 200000 for error and same weight cost)

Working on:

Kohonen (implemented, but dk how to test)
 - kept neighborhood and periodic boundary as default

OUT_DERIV_NOISE (should be working? no good way to test)


