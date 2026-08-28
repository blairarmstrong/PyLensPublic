import numpy as np
import matplotlib.pyplot as plt

output_cost_scale = 1
output_cost_strength = 1
ticks_per_interval = 1
min = 0
max = 1
p = 0.5

def linear_backward(outputs):
    """
    Computes the derivative of the linear cost function for a given set of outputs.

    Parameters:
        outputs (ndarray): The predicted values (outputs).

    Returns:
        ndarray: The computed derivatives for the linear cost function.
    """
    strength = output_cost_strength * output_cost_scale / ticks_per_interval
    if min is np.nan or max is np.nan:
        output_derivs = np.where(outputs > 0, strength, np.where(outputs < 0, -strength, 0.0))
    else:
        left_deriv = strength / (p - min)
        right_deriv = strength / (p - max)
        output_derivs = np.where(outputs < p, left_deriv, np.where(outputs > p, right_deriv, 0.0))
    return output_derivs

def conv_quad_backward(outputs):
    """
    Computes the derivative of the Convex Quadratic cost function for a given set of outputs.

    Parameters:
        outputs (ndarray)
        
    Returns:
        ndarray: The computed derivatives for the convex quadratic cost function.
    """
    min = p * 2 - 1 if p < 0.5 else 0.0
    scale = output_cost_strength * output_cost_scale * 2.0 / (ticks_per_interval * (p * p - min))
    output_derivs = (p - outputs) * scale
    return output_derivs

def cosine_backward(outputs):
    """
    Computes the derivative of the cosine cost function for a given set of outputs.

    Parameters:
        outputs (ndarray)

    Returns:
        ndarray: The computed derivatives for the cosine cost function.
    """
    strength = output_cost_strength * output_cost_scale * np.pi * 0.5 / ticks_per_interval
    invp = 1.0 / p
    inv1mp = 1.0 / (1.0 - p)
    output_derivs = np.where(outputs <= p, strength * invp * np.sin(np.pi * invp * outputs),
                            strength * inv1mp * np.sin(np.pi * inv1mp * (outputs - 2 * p + 1))
                            )
    return output_derivs

def logistic_backward(outputs):
    """
    Computes the derivative of the logistic cost function for a given set of outputs.

    Parameters:
        outputs (ndarray)

    Returns:
        ndarray: The computed derivatives for the logistic cost function.
    """
    strength = output_cost_strength * output_cost_scale / ticks_per_interval
    logp = np.log(p)
    log1mp = np.log(1.0 - p)
    scale = strength / logp if p <= 0.5 else strength / log1mp
    output_derivs = np.where(outputs < 1e-6, 1e-6, 
                                np.where(outputs > (1.0 - 1e-6), 1.0 - 1e-6, 
                                        (np.log(outputs) - logp - np.log(1-outputs) + log1mp) * scale))
    return output_derivs

def quadratic_backward(outputs):
    """
    Computes the derivative of the quadratic cost function for a given set of outputs.

    Parameters:
        outputs (ndarray)

    Returns:
        ndarray: The computed derivatives for the quadratic cost function.
    """
    strength = output_cost_strength * output_cost_scale * 2.0 / ticks_per_interval
    if min is np.nan or max is np.nan:
        output_derivs = strength * outputs
    else:
        left_scalue = strength / np.square(p - min)
        right_scale = - strength / np.square(max - p)
        output_derivs = np.where(outputs < p, left_scalue * (outputs - min), np.where(outputs > p, right_scale * (max - outputs), 0.0))
    return output_derivs

backward_map = {
    "linear": linear_backward,
    "quadratic": quadratic_backward,
    "conv_quad": conv_quad_backward,
    "logistic": logistic_backward,
    "cosine": cosine_backward
}

color_cycle = ['white', 'red', 'yellow', 'purple', 'blue']

def generate_derivs_plots():
    """
    Generates a grid of subplots showing the derivative curves for each cost function.
    """
    fig, axs = plt.subplots(2, 3)
    for idx, key in enumerate(backward_map.keys()):
        plt_coord = (idx // 3, idx % 3)
        outputs = [x / 100.0 for x in range(0, 101, 1)]
        derivs = []
        for output in outputs:
            output = np.array([output])
            derivs.append(backward_map[key](output))
            axs[plt_coord[0], plt_coord[1]].set_title(key)
        axs[plt_coord[0], plt_coord[1]].plot(outputs, derivs, label=key)
    plt.show()
    
def draw():
    """
    Draws a plot with derivative curves for each cost function, using different colors.
    """
    plt.gca().set_prop_cycle(plt.cycler('color', color_cycle))
    plt.gca().set_facecolor('black')
    for idx, key in enumerate(backward_map.keys()):
        outputs = [x / 100.0 for x in range(0, 101, 1)]
        derivs = []
        for output in outputs:
            output = np.array([output])
            derivs.append(backward_map[key](output))
        plt.plot(outputs, derivs, label=key)
    plt.legend(backward_map.keys(), loc="upper right")
    plt.show()
    
def main():
    draw()
    
if __name__ == "__main__":
    main()