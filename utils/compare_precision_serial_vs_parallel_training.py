# This is a random-mapping task, but a CONTINUOUS network has been thrown at it here.
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, ".")
from PyLens.simulator import Simulator

def rand_runner(parallel=False, extra_maxTime=False, num_worker=1):
    batch_size = 0
    report_interval = 1
    learning_rate = 0.5
    update_method = "dougs momentum"

    # create simulator
    sim_one = Simulator(name="simulator")

    # create sample network
    rand10x40 = sim_one.create_net(name='rand', time_intervals=4, ticks_per_interval=5,type='continuous')
    # rand10x40.plot = False
    rand10x40.toggle_plots(plots=False)
    rand10x40.toggle_keyboard(use=False)

    # create input layer with 10 units
    rand10x40.add_group(
        10,
        name="input",
        group_type="input",
        input_transforms=[],
        output_transforms=[]
    )

    # create hidden layer with 50 units; used sigmoid output, dot product input, and input integration
    rand10x40.add_group(
        3000,
        name="hidden",
        group_type="hidden",
        input_transforms=["dot", "in_integr"],
        output_transforms=["sigmoid",]
    )

    # create output layer with 10 units; used sigmoid output, dot product input, cross entropy error
    rand10x40.add_group(
        10,
        name="output",
        group_type="output",
        input_transforms=["dot"],
        output_transforms=["sigmoid", "out_integr"],
        error_function="cross_entropy"
    )

    # connects the input layer to the hidden layer; uniform links; full projection
    rand10x40.connect_groups(
        outgoing_group="input",
        incoming_group="hidden",
        initialization="uniform",
        proj_type="full"
    )

    # connects the output layer to the hidden layer; uniform links; full projection
    rand10x40.connect_groups(
        outgoing_group="output",
        incoming_group="hidden",
        initialization="uniform",
        proj_type="full"
    )


    # connects the hidden layer to the output layer; uniform links; full projection
    rand10x40.connect_groups(
        outgoing_group="hidden",
        incoming_group="output",
        initialization="uniform",
        proj_type="full"
    )


    # load example set
    if extra_maxTime:
        rand10x40.load_example_set("./examples/example_networks/rand10x40_examples/rand10x40.ex", def_s_max_time=10)
    else:
        rand10x40.load_example_set("./examples/example_networks/rand10x40_examples/rand10x40.ex")
    # load sample weights from clens random weight seed
    rand10x40.debug=True
    rand10x40.load_clen_weight("./utils/rand3000_for_debug.wt")

    # set learning rate and update method
    rand10x40.set_update_method(learning_rate, update_method)

    # to use GUI, uncomment
    # sim_one.use_gui(xor_net_one)

    if parallel == True:
        rand10x40.train(50, batch_size, report_interval, parallel_mode=True, num_worker=num_worker)
    else:
        rand10x40.train(50, batch_size, report_interval)
    return rand10x40.debug_errors, rand10x40.debug_weights
    

def find_divergence(l1, l2):
    for i, (a, b) in enumerate(zip(l1, l2)):
        if a != b:  # Check for exact divergence
            return i
    return None  # No divergence found

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print("Error: Missing required argument 'num_workers'.\n"
              "Usage: python compare_precision_serial_vs_parallel_training.py <num_workers>\n"
              "Please specify the number of workers as a command-line argument.")
        sys.exit(1)

    num_workers = sys.argv[1]
    errors_non_parallel, weights_non_parallel = rand_runner(extra_maxTime=False, parallel=False)
    errors_parallel, weights_parallel = rand_runner(extra_maxTime=False, parallel=True, num_worker=int(num_workers))


    error_divergence_index = find_divergence(errors_non_parallel, errors_parallel)
    weight_divergence_index = find_divergence(errors_non_parallel, errors_parallel)
    
    if error_divergence_index is None and weight_divergence_index is None:
        print('no divergence')
    else:
        print(f'errors divergence {error_divergence_index}')
        print(f'non parallel:{errors_non_parallel[error_divergence_index]}')
        print(f'non parallel:{errors_parallel[error_divergence_index]}')

        print(f'weights divergence {weight_divergence_index}')
        print(f'non parallel:{weights_non_parallel[weight_divergence_index]}')
        print(f'non parallel:{weights_parallel[weight_divergence_index]}')







