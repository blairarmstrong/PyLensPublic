Special Topics: Network Types
==============================

Standard Networks
-----------------

**Standard networks** are feed-forward networks in that information propagates all the way through the network during the forward pass on each tick. In such networks, groups are updated in the order in which they appear in the network's group array. A group update in this case consists of computing its inputs and immediately computing its outputs. This differs from continuous networks which have all groups update their inputs and then have all groups compute their outputs. Therefore, we say that continuous networks have synchronous update and standard network (including simple recurrent networks) have sequential update.

Simple Recurrent Networks and Simple Recurrent Backprop Through Time (SRBPTT) Network 
-------------------------

A **Simple Recurrent network (SRN)** is just a standard feed-forward network with one or more ``elman`` type groups. An ``elman`` context group is affiliated with a source group that has the same number of units. The context group should have the type ``elman``, but the source group can have any type.

When the context group computes its output, each unit increments its output by the output of the corresponding unit in the source group. If ``elman_clamp`` is the only output function, this essentially copies the outputs of the source group. Ordinarily, the context group should appear before the source group in the group order. In this case, the values copied will be the output of the source group from the previous tick (See :ref:`time-events-and-ticks`). Therefore, the context group is able to provide a bit of history.

It is customary to create a normal projection from the context group back to the source group, but this need not be the case. You might chain context groups together to get a history of length two or three ticks. If the source group is the output layer and the context group projects to a hidden layer, you will have a Jordan network.

During training, standard networks, including SRNs, perform a backpropagation sweep on each tick immediately after the forward pass. Continuous networks, on the other hand, perform a single backprop sweep that runs from the end of the example to the beginning of the example. You can extend the backpropagation phases of an SRN by increasing the backpropTicks parameter. A value of 3 will mean that, after each tick, the error will be backpropagated across the current tick and the previous two ticks. This can help SRNs learn long or difficult sequences. However, the training time of the network will increase in proportion to the backpropTicks.

An alternative is to use a **simple recurrent backprop through time (SRBPTT) network**. This is similar to an SRN in that it uses sequential updating. But it is similar to a continuous backprop through time network in that it uses a single backward sweep that runs from the end of the example to the beginning. It is like having the number of backpropTicks always equal to the number of ticks in the example, but much more efficient because there is just a single backward pass, rather than one for each tick. 

Fully Recurrent and Continuous Networks
---------------------------------------

**Fully recurrent networks** differ from simple recurrent networks in that fully recurrent networks use concurrent updates and propagate error derivatives backwards through time (See :ref:`time-events-and-ticks`). A feed-forward or simple recurrent network will transfer activity from the input layer to the output layer in a single tick: each group in order updates its inputs and immediately updates its outputs. In a fully recurrent network, on the other hand, all groups first update their inputs and then all groups update their outputs. Therefore, information can propagate across just one set of connections per tick.

What are typically called recurrent-backprop-through-time (RBPTT) networks use a single tick per time interval. Therefore, the unit activations will change completely on each tick. **Continuous networks** are a more general version of fully recurrent networks which use more than one tick per interval and integrate the unit inputs or outputs, causing them to change gradually. Because these are really part of a continuum, it's not always meaningful to draw a distinction between RBPTTs and continuous networks and they will collectively be called "fully recurrent" to distinguish them from simple recurrent networks.

Example events are the same in continuous, fully recurrent, and standard networks, each having an optional minTime, maxTime, and graceTime, with the example set defaults used when a value is not specified. These are specified in terms of time intervals, not in ticks, so the scale at which a continuous network is simulated can be changed without altering the example files. In the case of continuous networks, it becomes useful to have event time values that are not integers.

With continuous networks, the first tick of each example is used to set the initial outputs of the units. This tick therefore has no event associated with it. This is not necessary in standard networks because updates are sequential. 

Deterministic Boltzmann Machines
--------------------------------

**Deterministic Boltzmann machines (DBMs)** are typically fully interconnected networks that are trained using a two-phase settling process and a form of Hebbian weight updating. The input to a unit is the dot product of its incoming weights and the activations of the sending units. The units use a logistic transfer function (with an adjustable gain) and are output-integrating, meaning that the outputs change gradually over time. Groups in a DBM will use the ``BoltzmannInput`` input function and the ``BoltzmannOutput`` output function by default.

Training on each example event in a DBM occurs in two phases. At the start of the positive phase, both the input units and the output units are clamped to the inputs and targets, respectively, and the other units are reset to their ``initOutput``. Then the non-clamped units are allowed to settle for a given period of time or until the activations are no longer changing significantly. Usually the gain will be annealed with an exponential upward decay during the settling process. Initially there is high temperature (low gain), gradually settling to a lower temperature (high gain).

In the negative phase, the inputs remain clamped but the outputs are allowed to freely update. At the start of the negative phase, the activations of all of the units other than the inputs are shifted towards their ``initOutput``. Early in training, the outputs may not be shifted at all, and would thus remain the same as they were at the end of the positive phase. Gradually, the normalization could be made stronger until the units are completely reset to the ``initOutput`` at the start of the negative phase. During testing, only the negative phase is run.

During training, each weight is updated in a Hebbian fashion according to the products of the outputs of the preceding and following units. The weight change will be proportional to the difference between the product at the end of the positive phase and the product at the end of the negative phase. PyLens does not enforce the constraint that links in opposing directions between two units have the same weights, although they will receive the same changes so they will tend towards the same value if there is weight decay or will maintain a fixed offset otherwise. 
