# Filler task with srbptt (simple-recurrent-backprop-through-time) network
from PyLens.simulator import Simulator
import time
import numpy as np
import torch
import os



def run_backend(base_type, hidden_size):
    seed = 42
    batch_size = 0
    num_updates = 5
    report_interval = 1
    # Same initialization for both backends
    np.random.seed(seed)
    torch.manual_seed(seed)

    sim = Simulator(name="simulator", baseType=base_type)
    filler_net = sim.create_net(
        name="filler",
        time_intervals=5,
        type="srbptt",
    )

    filler_net.set_update_method(0.001, "adam")

    filler_net.add_group(
        4,
        name="chars",
        group_type="input",
        input_transforms=[],
        output_transforms=[],
    )

    filler_net.add_group(
        hidden_size,
        name="elman",
        group_type="elman",
        input_transforms=[],
        output_transforms=["sigmoid"],
    )

    filler_net.add_group(
        hidden_size,
        name="hidden",
        group_type="hidden",
        input_transforms=["dot"],
        output_transforms=["sigmoid"],
        lesion_rate=0,
        dropout_rate=0,
    )

    filler_net.add_group(
        20,
        name="output",
        group_type="output",
        input_transforms=["dot"],
        output_transforms=["sigmoid"],
        error_function="cross_entropy",
    )

    filler_net.connect_groups(
        outgoing_group="elman",
        incoming_group="hidden",
        link_type="uniform",
        proj_type="full",
    )

    filler_net.connect_groups(
        outgoing_group="chars",
        incoming_group="hidden",
        link_type="uniform",
        proj_type="full",
        lesion_rate=0,
        dropout_rate=0,
        perma_lesion_rate=0,
    )

    filler_net.connect_groups(
        outgoing_group="hidden",
        incoming_group="output",
        link_type="uniform",
        proj_type="full",
    )

    filler_net.connect_groups(
        outgoing_group="hidden",
        incoming_group="elman",
        link_type="elman",
        proj_type="one-to-one",
    )

    filler_net.load_example_set(
        "examples/lens_example_input/filler.ex"
    )

    weight_path = f'utils/benchmark_pytorch_numpy_weights_{hidden_size}.json'

    if os.path.exists(weight_path):
        filler_net.load_links(weight_path)
    else:
        filler_net.store_links(weight_path, weight_only=True, format='json')

    start = time.perf_counter()

    filler_net.train(
        epochs=num_updates,
        batch_size=batch_size,
        report_interval=report_interval,
    )

    elapsed = time.perf_counter() - start

    return elapsed


for hidden_size in [20, 500, 1000, 1500, 2000]:


    times = {}

    print(f"\nFiller with 1024 examples, hidden layer size: {hidden_size}")

    for base_type in ["numpy", "pytorch"]:
        print(f"\n{'=' * 20} {base_type.upper()} {'=' * 20}")
        times[base_type] = run_backend(base_type, hidden_size)


    print("Speed comparison")
    print("----------------")
    print(f"NumPy:   {times['numpy']:.3f} s")
    print(f"PyTorch: {times['pytorch']:.3f} s")

    if times["numpy"] < times["pytorch"]:
        print(f"NumPy is {times['pytorch'] / times['numpy']:.2f}x faster")
    else:
        print(f"PyTorch is {times['numpy'] / times['pytorch']:.2f}x faster")
