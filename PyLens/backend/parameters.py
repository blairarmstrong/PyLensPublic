from .array_factory import Array_factory as af

class NetworkParameters():
    """Defines network-wide hyperparameters for training and optimization."""
    def __init__(self): # default value
        """
        Initializes the network parameters with default values.
        
        Attributes:
            PAR_N_numTimeIntervals (int): Number of time intervals.
            PAR_N_numTicksPerInterval (int): Number of ticks per interval.
            PAR_N_backpropTicks (int): Number of ticks for backpropagation.
            PAR_N_numUpdates (int): Number of updates to perform.
            PAR_N_batchSize (int): Batch size for training.
            PAR_N_reportInterval (int): Interval for reporting training progress.
            PAR_N_criterion (float): Convergence criterion for training.
            PAR_N_trainGroupCrit (float): Criterion for training group.
            PAR_N_testGroupCrit (float): Criterion for testing group.
            PAR_N_groupCritRequired (bool): Whether group criterion is required.
            PAR_N_minCritBatches (int): Minimum number of batches for criterion.
            PAR_N_pseudoExampleFreq (bool): Whether pseudo-examples are used.
            PAR_N_algorithm (str): Optimization algorithm used.
            PAR_N_reset_on_example (bool): Whether to reset on each example.
            PAR_N_gain (float): Initial gain value.
            PAR_N_randRange (float): Range of random initialization values.
            PAR_N_noiseRange (float): Range of noise applied.
            PAR_N_clampStrength (float): Strength of clamping applied.
            PAR_N_initOutput (float): Initial output value.
            PAR_N_initOutputBias (float): Initial output bias.
            PAR_N_initInput (float): Initial input value.
            PAR_N_initOutputRange (float): Range for initial output values.
            PAR_N_ternaryShift (float): Shift factor for ternary values.
            PAR_N_initGain (float): Initial gain factor.
            PAR_N_finalGain (float): Final gain factor.
            PAR_N_annealTime (float): Annealing time parameter.
        """
        self.PAR_N_numTimeIntervals = 1 #1
        self.PAR_N_numTicksPerInterval = 1 #1
        self.PAR_N_backpropTicks = 1 #1
        self.PAR_N_numUpdates = 100 #3000
        self.PAR_N_batchSize = 0 # 0
        self.PAR_N_reportInterval = 10 #10
        self.PAR_N_criterion = 0.0 #0.0
        self.PAR_N_trainGroupCrit = 0.0 #0.0
        self.PAR_N_testGroupCrit = 0.0 #0.0
        self.PAR_N_groupCritRequired = False #False
        self.PAR_N_minCritBatches = 1 #1
        self.PAR_N_pseudoExampleFreq = False #False
        self.PAR_N_algorithm = "STEEPEST" #"STEEPEST"
        self.PAR_N_reset_on_example = True
        self.PAR_N_gain = 1.0  # 1.0
        self.PAR_N_randRange = 1.0 # 1.0
        self.PAR_N_noiseRange = 0.1 #0.1
        self.PAR_N_clampStrength = 0.5  # 0.5
        self.PAR_N_initOutput = 0.5  # 0.5
        self.PAR_N_initOutputBias = 1.0 # 1.0
        self.PAR_N_initInput = 0.0 # 0.0
        self.PAR_N_initOutputRange = 0.0  # 0.0
        self.PAR_N_ternaryShift = 5.0 # 5.0
        self.PAR_N_initGain = 1.0 #1.0
        self.PAR_N_finalGain = 1.0 #1.0
        self.PAR_N_annealTime = 1.0 #1.0

class OptimizerParameters():
    """Defines optimization-related parameters such as learning rate and momentum."""
    def __init__(self):
        """
        Initializes optimizer parameters with default values.
        
        Attributes:
            PAR_O_learningRate (float): Learning rate for optimization.
            PAR_O_momentum (float): Momentum factor.
            PAR_O_adaptiveGainRate (float): Adaptive gain rate.
            PAR_O_weightDecay (float): Weight decay factor.
            PAR_O_rate_increment (float): Learning rate increment factor.
            PAR_O_rate_decrement (float): Learning rate decrement factor.
            PAR_O_weightEliminationW0 (float): Initial weight elimination factor.
            PAR_O_gainDecay (float): Gain decay factor.
            PAR_O_outputCostStrength (float): Output cost strength.
            PAR_O_outputCostPeak (float): Peak value for output cost.
            PAR_O_targetRadius (float): Target radius.
            PAR_O_targetOneRadius (float): Radius for target one.
            PAR_O_targetZeroScaling (float): Scaling factor for zero target.
            PAR_O_zeroErrorRadius (float): Radius for zero error.
            PAR_O_AdamBeta_1 (float): Adam optimizer beta1 value.
            PAR_O_AdamBeta_2 (float): Adam optimizer beta2 value.
            PAR_O_AdamEpsilon (float): Epsilon value for Adam optimizer.
            PAR_O_gain (float): Gain factor.
            PAR_O_ternaryShift (float): Shift factor for ternary values.
            PAR_O_clampStrength (float): Strength of clamping.
            PAR_O_initOutput (float): Initial output value.
            PAR_O_initOutputRange (float): Range for initial output values.
            PAR_O_initInput (float): Initial input value.
            PAR_O_initGain (float): Initial gain value.
            PAR_O_finalGain (float): Final gain value.
            PAR_O_annealTime (float): Annealing time parameter.
            PAR_O_randMean (float): Mean for random initialization.
            PAR_O_randRange (float): Range for random values.
            PAR_O_noiseRange (float): Range of noise applied.
        """
        self.PAR_O_learningRate = 0.1 #0.2
        self.PAR_O_momentum = 0.9 #0.9
        self.PAR_O_adaptiveGainRate = 0.001 #0.001
        self.PAR_O_weightDecay = 0.0 #0.0
        self.PAR_O_rate_increment = 0.1 #0.1
        self.PAR_O_rate_decrement = 0.9 #0.9
        self.PAR_O_weightEliminationW0 = 0.0 #0.0
        self.PAR_O_gainDecay = 0.0 #0.0
        self.PAR_O_outputCostStrength = 0.01 #0.01
        self.PAR_O_outputCostPeak = 0.5 #0.5
        self.PAR_O_targetRadius = 0.0 #0.0
        self.PAR_O_targetOneRadius = 0.0  # 0.0
        self.PAR_O_targetZeroScaling = 1.0 #1.0
        self.PAR_O_zeroErrorRadius = 0.0 #0.0
        self.PAR_O_AdamBeta_1 = 0.9 #0.9
        self.PAR_O_AdamBeta_2 = 0.999 #0.999
        self.PAR_O_AdamEpsilon = 1e-8 #1e-8
        self.PAR_O_gain = 1.0 #1.0
        self.PAR_O_ternaryShift = 5.0 #5.0
        self.PAR_O_clampStrength = 0.5 #0.5
        self.PAR_O_initOutput = 0.5 #0.5
        self.PAR_O_initOutputRange = 0.0 #0.0
        self.PAR_O_initInput = 0.0 #0.0
        self.PAR_O_initGain = 1.0 #1.0
        self.PAR_O_finalGain = 1.0 #1.0
        self.PAR_O_annealTime = 1.0 #1.0
        self.PAR_O_randMean = 0.0 #0.0
        self.PAR_O_randRange = 1.0 #1.0
        self.PAR_O_noiseRange = 0.1 #0.1


class GroupParameters():
    """Defines parameters for groups in a neural network."""
    def __init__(self):
        """
        Initializes group-related parameters with default values.

        Attributes:
            PAR_G_standardReset (bool): Whether to use standard reset.
            PAR_G_continuousReset (bool): Whether to reset continuously.
            PAR_G_trainGroupCrit (float): Training group criterion value.
            PAR_G_testGroupCrit (float): Testing group criterion value.
            PAR_G_learningRate (float): Learning rate for group.
            PAR_G_momentum (float): Momentum for updates.
            PAR_G_weightDecay (float): Weight decay factor.
            PAR_G_weightEliminationW0 (float): Initial weight elimination parameter.
            PAR_G_gainDecay (float): Decay factor for gain.
            PAR_G_outputCostScale (float): Scaling factor for output cost.
            PAR_G_outputCostPeak (float): Peak output cost.
            PAR_G_targetRadius (float): Radius for target.
            PAR_G_targetZeroScaling (float): Scaling factor for zero target.
            PAR_G_zeroErrorRadius (float): Radius for zero error.
            PAR_G_errorScale (float): Error scaling factor.
            PAR_G_dtScale (float): Time step scaling factor.
            PAR_G_gain (float): Gain factor.
            PAR_G_ternaryShift (float): Shift value for ternary transformation.
            PAR_G_clampStrength (float): Strength of clamping.
            PAR_G_initOutput (float): Initial output value.
            PAR_G_initOutputRange (float): Range for initial output values.
            PAR_G_initInput (float): Initial input value.
            PAR_G_randMean (float): Mean for random initialization.
            PAR_G_randRange (float): Range of random values.
            PAR_G_noiseRange (float): Range of noise applied.
            PAR_G_noiseProc (str): Type of noise processing applied.
            PAR_G_showIncoming (bool): Whether to show incoming connections.
            PAR_G_showOutgoing (bool): Whether to show outgoing connections.
            PAR_G_numColumns (int): Number of columns in the group.
            PAR_G_neighborhood (int): Number of neighboring units.
            PAR_G_periodicBoundary (bool): Whether periodic boundary conditions apply.
        """
        self.PAR_G_standardReset = True #True
        self.PAR_G_continuousReset = True #True
        self.PAR_G_trainGroupCrit = af.nan #af.nan
        self.PAR_G_testGroupCrit = af.nan #af.nan
        self.PAR_G_learningRate = af.nan #af.nan
        self.PAR_G_momentum = af.nan #af.nan
        self.PAR_G_weightDecay = af.nan #af.nan
        self.PAR_G_weightEliminationW0 = af.nan #af.nan
        self.PAR_G_gainDecay = af.nan #af.nan
        self.PAR_G_outputCostScale = 1.0 #1.0
        self.PAR_G_outputCostPeak = af.nan #af.nan
        self.PAR_G_targetRadius = af.nan #af.nan
        self.PAR_G_targetZeroScaling = af.nan #af.nan
        self.PAR_G_zeroErrorRadius = af.nan #af.nan
        self.PAR_G_errorScale = 1.0 #1.0
        self.PAR_G_dtScale = 1.0 #1.0
        self.PAR_G_gain = af.nan #af.nan
        self.PAR_G_ternaryShift = af.nan #af.nan
        self.PAR_G_clampStrength = af.nan #af.nan
        self.PAR_G_initOutput = af.nan #af.nan
        self.PAR_G_initOutputRange = af.nan #af.nan
        self.PAR_G_initInput = af.nan #af.nan
        self.PAR_G_randMean = af.nan #af.nan
        self.PAR_G_randRange = af.nan #af.nan
        self.PAR_G_noiseRange = af.nan #af.nan
        self.PAR_G_noiseProc = "addGaussianNoise" #"addGaussianNoise"
        self.PAR_G_showIncoming = True #True
        self.PAR_G_showOutgoing = True #True
        self.PAR_G_numColumns = 0 #0
        self.PAR_G_neighborhood = 4 #4
        self.PAR_G_periodicBoundary = False #False

    def get_min_max_initOut(self, group): #from clens type.c function on line 281
        """
        Determines the minimum, maximum, and initial output values for a group.

        This function inspects the output transformations of a group and assigns appropriate
        min, max, and initial values based on predefined transformation types.

        Args:
            group (object): A group object containing `output_transforms` and `num_units`.

        Returns:
            list: A list containing three elements:
                - min_out (float): The minimum output value.
                - max_out (float): The maximum output value.
                - initOutput (float): The initial output value.
        """
        min_out = af.nan
        max_out = af.nan
        have_clamp = False
        initOutput = 0
        for output in group.output_transforms:
            if output in {"hard_clamp", "weak_clamp"}:
                have_clamp = True
            if output == "bias_clamp":
                min_out = 0
                max_out = 1
                initOutput = 1
            elif output in {"sigmoid", "gaussian", "LOGISTIC_ONCE", "soft_max", "KOHONEN", "OUT_BOLTZ", "OUT_NORM"}:
                min_out = 0 if (af.isnan(min_out) or 0 > min_out) else min_out
                max_out = 1 if (af.isnan(max_out) or 1 < max_out) else max_out
                initOutput = 1/group.num_units if output == "soft_max" else 0.5
            elif output in {"ternary", "tanh"}:
                min_out = -1 if (af.isnan(min_out) or -1 > min_out) else min_out
                max_out = 1 if (af.isnan(max_out) or 1 < max_out) else max_out
                initOutput = 0
            elif output == "exponential":
                min_out = 0 if (af.isnan(min_out) or 0 > min_out) else min_out
            elif output == "elman_clamp":
                pass
                #   unsure of what to do, below is C code type.c:281
                #     if ((S = (Group)P->otherData))
                #     {
                #     if (isNaN(min))
                #     min = S->minOutput;
                #     else
                #     min += S->minOutput;
                #     if (isNaN(max))
                #     max = S->maxOutput;
                #     else
                #     max += S->maxOutput;
                #     init = S->initOutput;
                #     }
                #     break;
                # }

        # if statment in C: if (isNaN(min) && isNaN(max) && (G->outputType & (CLAMPING_OUTPUT_TYPES & ~ELMAN_CLAMP)))
        # type.c: 281
        if af.isnan(min_out) and af.isnan(min_out): # if have_clamo is included it misses the input case which is init to 0.5
            min_out = 0
            max_out = 1
            initOutput = 0.5

        # Clenls only sets initOutput based on  if (!isNaN(G->initOutput)) type.c:343
        return [min_out, max_out, initOutput]




class LinkParameters():
    """Defines parameters for links (connections) in a neural network."""
    def __init__(self):
        self.PAR_L_max_weights = af.nan #af.nan
        self.PAR_L_min_weights = af.nan #af.nan
        self.PAR_L_learning_rate = 1 #1


class ErrorParamaters():
    """Defines parameters for error handling in training."""
    def __init__(self):
        self.PAR_E_large_value = 1e8
        self.PAR_E_small_value = 1e-8

class ExampleParameters():
    """Defines parameters for example processing in training."""
    def __init__(self):
        self.DEF_S_PIPELOOP = True
        self.DEF_S_MAXTIME = 1.0
        self.DEF_S_MINTIME = 0.0
        self.DEF_S_GRACETIME = 0.0
        self.DEF_S_DEFAULTINPUT = 0.0
        self.DEF_S_ACTIVEINPUT = 1.0
        self.DEF_S_DEFAULTTARGET = 0.0
        self.DEF_S_ACTIVETARGET = 1.0


"""
old defaults file 
"""

"""

import numpy as np


class NetworkDefaults():
    def __init__(self):
        self.DEF_N_numTimeIntervals = 1
        self.DEF_N_numTicksPerInterval = 1
        self.DEF_N_backpropTicks = 1
        self.DEF_N_numUpdates = 3000
        self.DEF_N_batchSize = 0
        self.DEF_N_reportInterval = 10
        self.DEF_N_criterion = 0.0
        self.DEF_N_trainGroupCrit = 0.0
        self.DEF_N_testGroupCrit = 0.5
        self.DEF_N_groupCritRequired = False
        self.DEF_N_minCritBatches = 1
        self.DEF_N_pseudoExampleFreq = False
        self.DEF_N_algorithm = "STEEPEST"


class OptimizerDefaults():
    def __init__(self):
        self.DEF_O_learningRate = 0.2
        self.DEF_O_momentum = 0.9
        self.DEF_O_adaptiveGainRate = 0.001
        self.DEF_O_weightDecay = 0.0
        self.DEF_O_rate_increment = 0.1
        self.DEF_O_rate_decrement = 0.9
        self.DEF_O_weightEliminationW0 = 0.0
        self.DEF_O_gainDecay = 0.0
        self.DEF_O_outputCostStrength = 0.01
        self.DEF_O_outputCostPeak = 0.5
        self.DEF_O_targetRadius = 0.0
        self.DEF_O_targetZeroScaling = 1.0
        self.DEF_O_zeroErrorRadius = 0.0
        self.DEF_O_AdamBeta_1 = 0.9
        self.DEF_O_AdamBeta_2 = 0.999
        self.DEF_O_AdamEpsilon = 1e-8
        self.DEF_O_gain = 1.0
        self.DEF_O_ternaryShift = 5.0
        self.DEF_O_clampStrength = 0.5
        self.DEF_O_initOutput = 0.5
        self.DEF_O_initOutputRange = 0.0
        self.DEF_O_initInput = 0.0
        self.DEF_O_initGain = 1.0
        self.DEF_O_finalGain = 1.0
        self.DEF_O_annealTime = 1.0
        self.DEF_O_randMean = 0.0
        self.DEF_O_randRange = 1.0
        self.DEF_O_noiseRange = 0.1


class GroupDefaults():
    def __init__(self):
        self.DEF_G_standardReset = True
        self.DEF_G_continuousReset = True
        self.DEF_G_trainGroupCrit = af.nan
        self.DEF_G_testGroupCrit = af.nan

        self.DEF_G_learningRate = af.nan
        self.DEF_G_momentum = af.nan
        self.DEF_G_weightDecay = af.nan

        self.DEF_G_weightEliminationW0 = af.nan
        self.DEF_G_gainDecay = af.nan
        self.DEF_G_outputCostScale = 1.0
        self.DEF_G_outputCostPeak = af.nan
        self.DEF_G_targetRadius = af.nan
        self.DEF_G_targetZeroScaling = af.nan
        self.DEF_G_zeroErrorRadius = af.nan
        self.DEF_G_errorScale = 1.0

        self.DEF_G_dtScale = 1.0
        self.DEF_G_gain = af.nan
        self.DEF_G_ternaryShift = af.nan
        self.DEF_G_clampStrength = af.nan
        self.DEF_G_initOutput = af.nan
        self.DEF_G_initOutputRange = af.nan
        self.DEF_G_initInput = af.nan

        self.DEF_G_randMean = af.nan
        self.DEF_G_randRange = af.nan
        self.DEF_G_noiseRange = af.nan
        self.DEF_G_noiseProc = "addGaussianNoise"

        self.DEF_G_showIncoming = True
        self.DEF_G_showOutgoing = True
        self.DEF_G_numColumns = 0
        self.DEF_G_neighborhood = 4
        self.DEF_G_periodicBoundary = False


class LinkDefaults():
    def __init__(self):
        self.DEF_L_max_weights = af.nan
        self.DEF_L_min_weights = af.nan
        self.DEF_L_learning_rate = 1

# DEF_U_target = af.nan
# DEF_U_externalInput = af.nan
# DEF_U_dtScale = 1.0
# DEF_U_activeTick = 1

"""

"""
old sanity checks

        network_defaults = NetworkDefaults()
        sanity_checks(self.PAR_N_numTimeIntervals, network_defaults.DEF_N_numTimeIntervals)
        sanity_checks(self.PAR_N_numTicksPerInterval, network_defaults.DEF_N_numTicksPerInterval)
        sanity_checks(self.PAR_N_backpropTicks, network_defaults.DEF_N_backpropTicks)
        sanity_checks(self.PAR_N_numUpdates, network_defaults.DEF_N_numUpdates)
        sanity_checks(self.PAR_N_batchSize, network_defaults.DEF_N_batchSize)
        sanity_checks(self.PAR_N_reportInterval, network_defaults.DEF_N_reportInterval)
        sanity_checks(self.PAR_N_criterion, network_defaults.DEF_N_criterion)
        sanity_checks(self.PAR_N_trainGroupCrit, network_defaults.DEF_N_trainGroupCrit)
        sanity_checks(self.PAR_N_testGroupCrit, network_defaults.DEF_N_testGroupCrit)
        sanity_checks(self.PAR_N_groupCritRequired, network_defaults.DEF_N_groupCritRequired)
        sanity_checks(self.PAR_N_minCritBatches, network_defaults.DEF_N_minCritBatches)
        sanity_checks(self.PAR_N_pseudoExampleFreq, network_defaults.DEF_N_pseudoExampleFreq)
        sanity_checks(self.PAR_N_algorithm, network_defaults.DEF_N_algorithm)
        
             optimizer_defaults = OptimizerDefaults()
        sanity_checks(self.PAR_O_learningRate, optimizer_defaults.DEF_O_learningRate)
        sanity_checks(self.PAR_O_momentum, optimizer_defaults.DEF_O_momentum)
        sanity_checks(self.PAR_O_adaptiveGainRate, optimizer_defaults.DEF_O_adaptiveGainRate)
        sanity_checks(self.PAR_O_weightDecay, optimizer_defaults.DEF_O_weightDecay)
        sanity_checks(self.PAR_O_rate_increment, optimizer_defaults.DEF_O_rate_increment)
        sanity_checks(self.PAR_O_rate_decrement, optimizer_defaults.DEF_O_rate_decrement)
        sanity_checks(self.PAR_O_weightEliminationW0, optimizer_defaults.DEF_O_weightEliminationW0)
        sanity_checks(self.PAR_O_gainDecay, optimizer_defaults.DEF_O_gainDecay)
        sanity_checks(self.PAR_O_outputCostStrength, optimizer_defaults.DEF_O_outputCostStrength)
        sanity_checks(self.PAR_O_outputCostPeak, optimizer_defaults.DEF_O_outputCostPeak)
        sanity_checks(self.PAR_O_targetRadius, optimizer_defaults.DEF_O_targetRadius)
        sanity_checks(self.PAR_O_targetZeroScaling, optimizer_defaults.DEF_O_targetZeroScaling)
        sanity_checks(self.PAR_O_zeroErrorRadius, optimizer_defaults.DEF_O_zeroErrorRadius)
        sanity_checks(self.PAR_O_AdamBeta_1, optimizer_defaults.DEF_O_AdamBeta_1)
        sanity_checks(self.PAR_O_AdamBeta_2, optimizer_defaults.DEF_O_AdamBeta_2)
        sanity_checks(self.PAR_O_AdamEpsilon, optimizer_defaults.DEF_O_AdamEpsilon)
        sanity_checks(self.PAR_O_gain, optimizer_defaults.DEF_O_gain)
        sanity_checks(self.PAR_O_ternaryShift, optimizer_defaults.DEF_O_ternaryShift)
        sanity_checks(self.PAR_O_clampStrength, optimizer_defaults.DEF_O_clampStrength)
        sanity_checks(self.PAR_O_initOutput, optimizer_defaults.DEF_O_initOutput)
        sanity_checks(self.PAR_O_initOutputRange, optimizer_defaults.DEF_O_initOutputRange)
        sanity_checks(self.PAR_O_initInput, optimizer_defaults.DEF_O_initInput)
        sanity_checks(self.PAR_O_initGain, optimizer_defaults.DEF_O_initGain)
        sanity_checks(self.PAR_O_finalGain, optimizer_defaults.DEF_O_finalGain)
        sanity_checks(self.PAR_O_annealTime, optimizer_defaults.DEF_O_annealTime)
        sanity_checks(self.PAR_O_randMean, optimizer_defaults.DEF_O_randMean)
        sanity_checks(self.PAR_O_randRange, optimizer_defaults.DEF_O_randRange)
        sanity_checks(self.PAR_O_noiseRange, optimizer_defaults.DEF_O_noiseRange)


 group_defaults = GroupDefaults()
        sanity_checks(self.PAR_G_standardReset, group_defaults.DEF_G_standardReset)
        sanity_checks(self.PAR_G_continuousReset, group_defaults.DEF_G_continuousReset)
        sanity_checks(self.PAR_G_trainGroupCrit, group_defaults.DEF_G_trainGroupCrit)
        sanity_checks(self.PAR_G_testGroupCrit, group_defaults.DEF_G_testGroupCrit)
        sanity_checks(self.PAR_G_learningRate, group_defaults.DEF_G_learningRate)
        sanity_checks(self.PAR_G_momentum, group_defaults.DEF_G_momentum)
        sanity_checks(self.PAR_G_weightDecay, group_defaults.DEF_G_weightDecay)
        sanity_checks(self.PAR_G_weightEliminationW0, group_defaults.DEF_G_weightEliminationW0)
        sanity_checks(self.PAR_G_gainDecay, group_defaults.DEF_G_gainDecay)
        sanity_checks(self.PAR_G_outputCostScale, group_defaults.DEF_G_outputCostScale)
        sanity_checks(self.PAR_G_outputCostPeak, group_defaults.DEF_G_outputCostPeak)
        sanity_checks(self.PAR_G_targetRadius, group_defaults.DEF_G_targetRadius)
        sanity_checks(self.PAR_G_targetZeroScaling, group_defaults.DEF_G_targetZeroScaling)
        sanity_checks(self.PAR_G_zeroErrorRadius, group_defaults.DEF_G_zeroErrorRadius)
        sanity_checks(self.PAR_G_errorScale, group_defaults.DEF_G_errorScale)
        sanity_checks(self.PAR_G_dtScale, group_defaults.DEF_G_dtScale)
        sanity_checks(self.PAR_G_gain, group_defaults.DEF_G_gain)
        sanity_checks(self.PAR_G_ternaryShift, group_defaults.DEF_G_ternaryShift)
        sanity_checks(self.PAR_G_clampStrength, group_defaults.DEF_G_clampStrength)
        sanity_checks(self.PAR_G_initOutput, group_defaults.DEF_G_initOutput)
        sanity_checks(self.PAR_G_initOutputRange, group_defaults.DEF_G_initOutputRange)
        sanity_checks(self.PAR_G_initInput, group_defaults.DEF_G_initInput)
        sanity_checks(self.PAR_G_randMean, group_defaults.DEF_G_randMean)
        sanity_checks(self.PAR_G_randRange, group_defaults.DEF_G_randRange)
        sanity_checks(self.PAR_G_noiseRange, group_defaults.DEF_G_noiseRange)
        sanity_checks(self.PAR_G_noiseProc, group_defaults.DEF_G_noiseProc)
        sanity_checks(self.PAR_G_showIncoming, group_defaults.DEF_G_showIncoming)
        sanity_checks(self.PAR_G_showOutgoing, group_defaults.DEF_G_showOutgoing)
        sanity_checks(self.PAR_G_numColumns, group_defaults.DEF_G_numColumns)
        sanity_checks(self.PAR_G_neighborhood, group_defaults.DEF_G_neighborhood)
        sanity_checks(self.PAR_G_periodicBoundary, group_defaults.DEF_G_periodicBoundary)


        link_defaults = LinkDefaults()
        sanity_checks(self.PAR_L_max_weights, link_defaults.DEF_L_max_weights)
        sanity_checks(self.PAR_L_min_weights, link_defaults.DEF_L_min_weights)
        sanity_checks(self.PAR_L_learning_rate, link_defaults.DEF_L_learning_rate)


def sanity_checks(value_set, default_value):
    if value_set is None:
        value_set = default_value

"""
