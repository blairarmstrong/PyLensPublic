# This is a random-mapping task, but a CONTINUOUS network has been thrown at it here.
import sys
sys.path.insert(0, ".")
from PyLens.simulator import Simulator

def rand_runner(parallel=False, extra_maxTime=False):
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
        50,
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
        link_type="uniform",
        proj_type="full"
    )

    # connects the output layer to the hidden layer; uniform links; full projection
    rand10x40.connect_groups(
        outgoing_group="output",
        incoming_group="hidden",
        link_type="uniform",
        proj_type="full"
    )


    # connects the hidden layer to the output layer; uniform links; full projection
    rand10x40.connect_groups(
        outgoing_group="hidden",
        incoming_group="output",
        link_type="uniform",
        proj_type="full"
    )


    # load example set
    if extra_maxTime:
        rand10x40.load_example_set("./examples/example_networks/rand10x40_examples/rand10x40.ex", def_s_max_time=10)
    else:
        rand10x40.load_example_set("./examples/example_networks/rand10x40_examples/rand10x40.ex")
    # load sample weights from clens random weight seed
    rand10x40.load_clen_weight("./examples/example_networks/rand10x40_examples/rand10x40_weights.wt")

    # set learning rate and update method
    rand10x40.set_update_method(learning_rate, update_method)

    # to use GUI, uncomment
    # sim_one.use_gui(xor_net_one)

    if parallel == True:
        rand10x40.train(10, batch_size, report_interval, parallel_mode=True, num_worker=3)
    else:
        rand10x40.train(10, batch_size, report_interval)

    if extra_maxTime:
        clens_error = 1231.74
        clens_weight_cost = 257.513
    else:
        clens_error = 285.14
        clens_weight_cost = 260.360

    pylens_error = round(rand10x40.stats_plotter.progress_stats['error'][-1], 2)
    pylens_weight_cost = round(rand10x40.stats_plotter.progress_stats["weight_cost"][-1], 3)

    assert clens_error == pylens_error
    assert clens_weight_cost == pylens_weight_cost

if __name__ == '__main__':
    rand_runner(extra_maxTime=False)
    rand_runner(extra_maxTime=True)
    rand_runner(extra_maxTime=False, parallel=True)
    rand_runner(extra_maxTime=True, parallel=True)











