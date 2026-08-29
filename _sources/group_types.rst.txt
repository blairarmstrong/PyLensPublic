Special Topics: Group Types
==============================

This documentation is adapted from the original Lens documentation.

Group (layer) types are divided into four classes. There are the basic types, input types, output types, and cost types. The basic types are simply a set of flags that include the basic types, such as ``input``, ``output``, ``bias``, and ``elman``, and any other types that don't happen to fit the other categories.

The input, output, and cost types, on the other hand, actually reflect particular transformation functions. These functions will be executed in the order in which the types were specified. 


Basic Types
------------

The basic type is the primary type of the units in a group. The main effect of the basic type is that it determines the default input, output, and cost functions of the group. The process of determining default functions can be fairly complicated as it depends on network type and other group types. So if you don't want to figure it all out, you may want to just check the types of your groups from the attribute ``group_type`` of ``group`` to make sure types are assigned as you expect and then explicitly specify them as necessary.

.. Note that there is no HIDDEN type. A hidden group simply has no basic type. The default input type for a hidden group is DOT_PRODUCT. The default output type is LOGISTIC. However, if the network is a BOLTZMANN net, the default input type is IN_BOLTZ and the default output type is OUT_BOLTZ. Hidden groups will automatically receive an incoming link from the bias unit. This can be prevented with the -BIASED type, as explained below.

``bias``
    A bias group is one that always maintains an output of 1.0. Ordinarily, a bias group will be created automatically with the network and there is probably no need for more than one, so you may never need to use this type. BIAS cannot be combined with any other basic type. A bias group by default is itself unbiased, has no input type, and has a ``bias_clamp`` output type. The ``bias_clamp`` function simply sets the output to 1.0. 

``input``
    The main implication of the ``input`` type is that ``external_input`` will be assigned to units in the group when an example event is loaded. Any group which is expecting external inputs should be of type ``input``. 

``output``
    The main implication of the ``output`` type is that targets will be assigned to units in the group when an example event is loaded. These groups are treated the same as hidden groups. However, ``output`` groups will also be assigned a default error function. Groups with ``Kohonen`` or ``BoltzmannOutput`` output transformations will have no default error procedure. ``linear`` groups will have ``SquaredError`` error. ``Soft_Max`` groups will have ``Divergence`` error. All other groups will have CROSS_ENTROPY as their default error function.

``elman``
    The ``elman`` type can be used to identify context groups in simple recurrent networks. Like ``bias`` and ``input`` groups, ``elman`` groups have no default bias inputs and no default input function. The default output function is ``elman_clamp``. Before an elman_clamped group is functional, you must use ``elman_connect`` to assign a source group to it. 


Input Transformations types
------------------------------

The input transformations form a pipeline of functions which compute the units' inputs in the forward direction and backpropagate the ``input_derivs`` in the backward direction. The basic input transformations actually compute a function over incoming links to produce an input value. The other input transformations modify this value. There shouldn't be more than one transformation and there must be one if the group's input will ever be used, which may not be the case for ``input`` or ``elman`` groups.

Basic Input Transfromation types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Dot_Product``
    This is by far the most common input type. It computes the dot product of the incoming weight vector with the incoming activation vector. In other words, it takes the sum over all incoming links of the product of the link weight and the output of the unit the link is coming from. This is the default unless the network is a Boltzmann machine. 

``Distance``
    This computes the squared distance between the weight vector and the activation vector. In other words, it takes the sum over all incoming links of the square of the difference between the link weight and the output of the unit the link is coming from. You probably do not want to use this for a backpropagation network. It should be used along with the ``Kohonen`` output type for Kohonen networks. 

``Product``
    This takes the product of all incoming weights and the outputs of the units from which they come. This can be used to perform the Pi part of a Sigma-Pi unit, which is actually implemented in Lens using more than one unit. ``Product`` could be used for the gating part of a gated unit, which is essentially a Pi-Sigma. Often the weights involved will actually be frozen at 1.0 so that only the sending unit activations are relevant. 

``BoltzmannInput``
    This is the input half of a Boltzmann unit. If the unit is clamped by either an ``external_input`` or a target within the grace period (see ``BoltzmannOutput``), this does nothing. Otherwise, it computes a dot product.  In the backward pass, this does not propagate the usual derivatives to the sending units. Instead, it increments the incoming link derivatives by computing an outer product between the receiving-group outputs and the sending-group outputs:

    ``receiving_output_matrix[:, None] @ sending_output_matrix[None, :] - receiving_output_derivs[:, None] @ sending_output_derivs[None, :].``

    This produces a matrix with the same shape as the link weight matrix. Each entry gives the update for one connection from a sending unit to a receiving unit. ``output_matrix`` stores the negative-phase outputs. ``output_derivs`` is filled after the positive-phase by caching the current output_matrix. 

``In_Copy``
    The units in a group with an ``In_Copy`` input function simply copy their inputs from some field in the corresponding units of another group. The ``copy_connect`` command must be used to specify which group and which field will be the source of the copying. 

Input Modifying Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^

``Soft_Clamp``
    The ``Soft_Clamp`` function assumes that the output function is logistic. It adds a factor to the input of the unit such that, with no other input, the output of the unit would be:

    ``init_output + clamp_strength * (external_input - init_output)``

    Thus, the output would fall between the ``init_output`` and the ``external_input``. The ``clamp_strength``, which ranges from 0.0 to 1.0, determines the extent to which the output will be dominated by the ``external_input``. This is meant to be used by groups that also receive ordinary inputs. The ``clamp_strength`` should be less than 1.0 if the ``external_inputs`` are 0.0 or 1.0 or the group will have infinite input. 

``IncrementClamp``
    This function simply adds the ``external_input``, scaled by the ``clamp_strength``, to the unit's input. It is used in interactive activation models, among other things. 

``In_Integr``
    This time-averages the group's input according to the function:

    ``lastinput += dt * (input_matrix - lastinput)``

    It is ordinarily used with ``continuous`` networks. With a ``sigmoid`` output function, it differs from ``Out_Integr`` in that units will adapt more rapidly when being pulled toward the extremes and less rapidly when being pulled towards an output of 0.5. 


Output Transformations
------------

The group output transformations (also called activation functions) form a pipeline of functions which compute the units' outputs in the forward direction and backpropagate the ``output_derivs`` in the backward direction. The basic transformation determine the output as a function of the input. The clamping transformation set or alter the output based on the ``external_input``. The other transformations modify an already-computed output value. There shouldn't be more than one basic transformation. There may be no basic transformation if there is a clamping transformation.

Basic Output transformations
^^^^^^^^^^^^^^^^^^

``Linear``
    This simply copies the input to the output. 

``Sigmoid``
    Computes the traditional sigmoid function:

    O = 1 / (1 + exp(-i * gain))

    Gain is the inverse of the temperature. It is used to avoid division. Ordinarily the gain is taken from the network's gain field or the group's gain field if that is set.

``Ternary``
    This is essentially a normal sigmoid shifted to the right added to a negated sigmoid of -i shifted to the left. Alternately, you can think of it as a [-1,1] sigmoid that has a flat place at 0. It is designed to give the unit stable outputs at -1, 1, and 0. You could think of such units as coding whether a feature is present, absent, or unknown. The gain affects the slope of each of the two sigmoids. The ternaryShift sets the distance between their centers. Increasing the ternaryShift will make the central plateau wider. Increasing the gain will make the transitions between plateaus sharper. 

``Tanh``
    This is equivalent to 1 - 2S(2 i), where S is the ordinary sigmoid function and I is the input. Note that its slope is actually twice what the slope would be if you just stretched a sigmoid to the range [-1,1]. So you may want to use half the normal gain to compensate.

``Exponential``
    This is just exp(i). There is a big potential for overflow with this, so you may want to be careful how you use it. 

``Gaussian``
    This computes a gaussian radial basis function: exp(-i^2 * gain^2). This is often as effective as ``Sigmoid``, although it can become a bit unstable at the end of training.

``SoftMax``
    This is equivalent to an exponential followed by a normalization. However, ``SoftMax`` scales the values before computing the exponential. This doesn't affect the end result but it avoids overflow. 
    
    .. A ``SoftMax`` output group will get DIVERGENCE error by default. 

``Kohonen``
    This is used for the map layer in a ``Kohonen`` network. It should be combined with a ``Distance`` input function. It finds the unit whose weight vector is most similar to the input vector. Any unit in the map whose squared Euclidean distance from the best unit is greater than neighborhood will be silent. Groups in the neighborhood will have output equal to 1.0 minus the ratio between the unit's input and the largest input of any unit. The output will therefore fall in the range [0.0, 1.0].

    In the backward pass, the ``input_deriv`` of units in the neighborhood will be set to 1.0 and that of the others to 0.0. Only units in the neighborhood will be able to alter their incoming weights. The ``Distance`` procedure, in the backward pass, will cause the incoming weights to drift towards the input vector. 

``BoltzmannOutput``
    This is used for groups in a Boltzmann network. If the unit has an ``external_input``, the output will be clamped to that value. Otherwise, if it has a target and the network is in the ``in_grace_period`` (the positive phase of the Boltzmann algorithm), the output will be clamped to the target. Otherwise, it is computed as a time-averaged sigmoid of the input. 

``Out_Copy``
    The units in a group with an ``Out_Copy`` output function simply copy their outputs from some field in the corresponding units of another group. The ``copy_connect`` command must be used to specify which group and which field will be the source of the copying. 

``Interact_Integr``
    This implements the interactive-activation output rule. For a traditional interactive-activation model, it should be used with ``Dot_Product`` inputs, and an ``IncrementClamp`` input function. It contains a decay term and time-averages the activations. The decay is fixed at 1.0. This version crops the unit outputs to the range [0, maxOutput] because negative outputs are not normally used in an IA model. 

Clamping Output Transformations
^^^^^^^^^^^^^^^^^^^^^

``Hard_Clamp``
    If the ``external_input`` is a real number, this sets the output to the ``external_input``. Otherwise it does nothing. 

``Bias_Clamp``
    This sets the output to the initOutput (defaults to 1.0 for BIAS groups). 

``Elman_Clamp``
    In order for an ``Elman_Clamp`` function to work, you must first use ``elman_connect`` to associate a source group with the context group. This simply copies the (cached) output from each source unit and adds it to the output of the corresponding context unit. It is possible to have more that one ``Elman_Clamp`` function. In this case, the output will simply sum the outputs from each of the source units. If a group has multiple ``Elman_Clamp`` functions, each call to ``elman_connect`` will define the source group for the first function that has not yet been assigned a group. 

``Weak_Clamp``
    This shifts the output a certain fraction of the way towards the ``external_input``. The fraction is determined by the ``clamp_strength``. Specifically, the function is: ``o = o + clamp_strength * (external_input - o)``

Output Modifying Transformations
^^^^^^^^^^^^^^^^^^^^^^^

``Out_Integr``
    This is just like ``In_Integr`` but it integrates the output rather than the input. 
    
    .. This is put on by default in a CONTINUOUS network unless IN_INTEGR is specified. 

``Out_Norm``
    This normalizes the outputs of the units in the group to sum to 1.0. It probably should not be used unless the un-normalized values are constrained to be positive. The ``SoftMax`` function should be used rather than an ``Exponential`` followed by ``Out_Norm`` because ``SoftMax`` will avoid numerical overflow. 
``Noise``
    This makes the output noisy. The type of noise is determined by the group's ``noise_proc`` and ``noise_range`` parameters. 

``Out_Deriv_Noise``
    This injects noise into the outputDerivs on the backward pass. The type of noise is determined by the group's ``noise_proc`` and ``noise_range`` parameters.

``Out_Cropped``
    This crops the output to within the range [``minOutput``, ``maxOutput``]. You may want to use this after ``Out_Noise`` to prevent outputs outside of this range. 

``Out_Winner``
    This is a winner-take-all filter. The most active unit retains its activation and the other units are set to the minimum output value for the group. In the backward phase, the original outputs are restored to enable error to be backpropagated across the transfer function. 

Cost Types
----------

 There are actually two very different kinds of cost functions: error functions and unit output cost functions. The error functions are based on the similarity of the outputs and targets. The unit output cost functions simply charge the unit for producing certain outputs, such as non-binary ones. The error functions assess no error when the target is NaN.

Error Types
^^^^^^^^^^^^

``SquaredError``
    This simply takes the sum over all units of the squared difference between the output and target. This is only the default for ``Linear`` output groups. 

``CrossEntropyError``
    This is the sum over all units of:

    t log(t/o) + (1-t) log((1-t)/(1-o)),

    where t is the target and o is the output. This can become infinite if the output incorrectly reaches 0.0 or 1.0. This may happen if the training parameters are too aggressive. Lens caps the error at a very large value. ``CrossEntropyError`` is the default error type for most output groups. 

``DivergenceError``
    This is the sum over all units of:
    t log(t/o) This is only stable if the target vector and output vector are each normalized to sum to 1.0. This is the default error type for ``Soft_Max`` output groups. 

``CosineError``
    This calculates the 1.0 - the cosine of the angle between the output and target vectors. This can be used for training as well as evaluation. However, training can be tricky because there is only pressure for the angle of the output vector to be correct, not the absolute values of the outputs. You could use a unit cost function (such as ``LogisticCost``) on the output units to encourage them to be binary if that is desired. 

.. Target Types
.. ^^^^^^^^^^^^

.. TARGET_COPY
..     The units in a group with a TARGET_COPY cost function will copy their targets from some field in the corresponding units of another group. The copyConnect command must be used to specify which group and which field will be the source of the copying. The TARGET_COPY type should be specified prior to the main error type. 

Unit Cost Types
^^^^^^^^^^^^^^^^^

Unit output costs are error terms that penalize units for having certain outputs. For bounded units (ones whose outputs are limited to a finite range), there are five unit cost functions, all of which encourage the unit to have binary output. Non-bounded units can have one of two cost functions that encourage the unit to be silent. Output costs would typically only be applied to hidden layers, although they may be useful on output layers as well. They can be used with simple and continuous networks, but not with Boltzmann machines.

When used on a bounded group, the cost functions will be low at the extremes and will have a maximum cost of 1.0 at the outputCostPeak, which is typically at 0.5.

``LinearCost``
    For a bounded unit this changes linearly from 1.0 at the peak to 0.0 at the min and max output. For an unbounded unit, this is simply equal to the absolute value of the output. 

``QuadraticCost``
    For a bounded unit, this has a derivative of 0 at the extremes and slopes up concavely to the peak. For unbounded units this is equal to the output squared. 

``ConvQuadCost``
    This can only be used on bounded units. It is shaped like a downward-facing parabola. The derivative is 0 at the peak. 

``LogisticCost``
    This can only be used on bounded units. It is similar in shape to the ``ConvQuadCost`` but the derivative goes to infinity as it approaches the extremes. However, the derivative is capped as if the output could not get closer than 1e-6 of the min or max. 

``CosineCost``
    This can only be used on bounded units. It has zero derivative at the min, max, and the peak. 
