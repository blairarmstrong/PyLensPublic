import sys
sys.path.insert(0, ".")
from PyLens.simulator import Simulator


def task_test():
    batch_size = 0
    num_updates = 10 
    report_interval = 1
    learning_rate = 0.01
    update_method = "dougs momentum"

    # create simulator
    sim_one = Simulator(name="simulator")

    # create sample network
    task_net_one = sim_one.create_net(name="task")

    task_net_one.plot = False

    # create input layer with 3 units
    task_net_one.add_group(
        3,
        name="task",
        group_type="input",
        input_transforms=[],
        output_transforms=[]
    )

    # create hidden layer with 6 units; used sigmoid output, dot product input
    task_net_one.add_group(
        6,
        name="taskHidden",
        group_type="hidden",
        input_transforms=["dot"],
        output_transforms=["sigmoid"]
    )

    # create input layer with 6 units; used sigmoid output, dot product input
    task_net_one.add_group(
        6,
        name="input",
        group_type="input",
        input_transforms=["dot"],
        output_transforms=["sigmoid"]
    )

    # create input layer with 6 units; used sigmoid output, product input
    task_net_one.add_group(
        6,
        name="hidden",
        group_type="hidden",
        input_transforms=["product"],
        output_transforms=["sigmoid"]
    )

    # create output layer with 2 units; used sigmoid output, dot product input
    task_net_one.add_group(
        2,
        name="output",
        group_type="output",
        input_transforms=["dot"],
        output_transforms=["sigmoid"]
    )


    # connects the task layer to the taskHidden layer; uniform links; full projection
    task_net_one.connect_groups(
        outgoing_group="task",
        incoming_group="taskHidden",
        initialization="uniform",
        proj_type="full"
    )

    # connects the taskHidden layer to the hidden layer; uniform links; full projection
    task_net_one.connect_groups(
        outgoing_group="taskHidden",
        incoming_group="hidden",
        initialization="uniform",
        proj_type="one-to-one"
    )

    # connects the input layer to the hidden layer; uniform links; full projection
    task_net_one.connect_groups(
        outgoing_group="input",
        incoming_group="hidden",
        initialization="uniform",
        proj_type="one-to-one"
    )

    # connects the hidden layer to the output layer; uniform links; full projection
    task_net_one.connect_groups(
        outgoing_group="hidden",
        incoming_group="output",
        initialization="uniform",
        proj_type="full"
    )

    # load example set
    task_net_one.load_example_set("./examples/lens_example_input/task.ex")

    # load sample weights from clens random weight seed
    task_net_one.load_clen_weight("./examples/example_networks/task_examples/task_example_one_weights.wt")

    # set learning rate and update method
    task_net_one.set_update_method(learning_rate, update_method)

    # to use GUI, uncomment
    # sim_one.use_gui(task_net_one)

    task_net_one.train(num_updates, batch_size, report_interval)

    # COMPARE ERRORS AND WEIGHT COST

    clens_error = round(421.972, 3)
    clens_weight_cost = round(9.16299, 3)
    clens_grad_linearity = round(0.99742, 3)

    # pylens - error:  weight cost:
    pylens_error = round(task_net_one.stats_plotter.progress_stats['error'][-1], 3)
    pylens_weight_cost = round(task_net_one.stats_plotter.progress_stats["weight_cost"][-1], 3)
    pylens_gradient_linearity = round(task_net_one.stats_plotter.progress_stats["grad_lin"][-1], 3)

    assert pylens_error == clens_error, "Errors differ by:{}".format(abs(pylens_error-clens_error))
    assert pylens_weight_cost == clens_weight_cost, "Weight costs differ by: {}".format(
        abs(pylens_weight_cost-clens_weight_cost))

    print("testing task:")
    print("     Error: ... ", end="")
    if pylens_error != clens_error:
        print("Failed")
        print("Errors differ by: {}".format(abs(pylens_error-clens_error)))
        return False
    print("passed")
    print("     Weight Costs: ... ", end="")
    if pylens_weight_cost != clens_weight_cost:
        print("Failed")
        print("Weight costs differ by: {}".format(abs(pylens_weight_cost-clens_weight_cost)))
        return False
    print("passed")
    print("     Gradient Linearity: ... ", end="")
    if pylens_weight_cost != clens_weight_cost:
        print("Failed")
        print("Gradient Linearity differ by: {}".format(abs(pylens_gradient_linearity-clens_grad_linearity)))
        return False
    print("passed")
    return True

result = task_test()

if result:
    print("Final Passed")
    exit(0)
else:
    print("Final Failed")
    exit(1)










