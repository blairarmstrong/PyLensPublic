from PyLens.backend.array_factory import Array_factory as af
from PyLens.backend.boltzmann_network import BoltzmannMachine

from PyLens.backend.network import Network
from PyLens.backend.continuous_network import ContinuousNetwork
from PyLens.backend.srbptt_network import SRBPTTNetwork
from PyLens.gui.unit_viewer_tk import FrameExamplesProgram
from PyLens.gui.link_viewer_tk import link_viewer
from PyLens.gui.main_viewer_tk import main_viewer_tk

# Unit viewer layout builder (Lens-style plotRow)
from PyLens.gui.unit_layout import build_layout
import threading
from subprocess import Popen, PIPE

from queue import Queue
from queue import Empty, Full, Queue
import time
from threading import Thread
from PyLens.backend.parameters import NetworkParameters
network_params = NetworkParameters()


class Simulator:

    def __init__(self, name="simulator", baseType='numpy', parallel_mode=True):
        self.name = name
        self.baseType = baseType
        af.set_base_type(baseType)
        self.parallel_mode = parallel_mode
        self.networks = []
        self.active_network = None
        self.gui_program = None
        # for now we assume one Simulator has exactly one Network
        self.saved_history = None
        
    def use_net(self, net):
        for network in self.networks:
            if network.name == net:
                self.active_network = network
                return True
        return False

    def create_net(self, name, time_intervals=1, ticks_per_interval=1, learning_rate=0.1, add_bias=True, type='Standard'):

        if type == 'continuous':
            net = ContinuousNetwork(name=name, baseType=self.baseType, time_intervals=time_intervals, ticks_per_interval=ticks_per_interval, learning_rate=learning_rate, add_bias=add_bias)
        elif type == 'srbptt':
            net = SRBPTTNetwork(name=name, baseType=self.baseType, time_intervals=time_intervals, ticks_per_interval=ticks_per_interval, learning_rate=learning_rate, add_bias=add_bias)
        elif type == 'boltzmann':
            net = BoltzmannMachine(name=name, baseType=self.baseType, time_intervals=time_intervals, ticks_per_interval=ticks_per_interval, learning_rate=learning_rate, add_bias=add_bias)
        elif type == 'Standard':
            net = Network(name=name, baseType=self.baseType, time_intervals=time_intervals, ticks_per_interval=ticks_per_interval, learning_rate=learning_rate, add_bias=add_bias)
        else:
            print(f"Error: The network type '{type}' is not supported.")
            exit(-1)
            return
        return net
    
    def set_time(self, ticks_per_interval=None, time_intervals=None):
        if ticks_per_interval is not None:
            self.active_network.ticks_per_interval = ticks_per_interval
            self.active_network.dt = 1 / ticks_per_interval
        if time_intervals is not None:
            self.active_network.time_intervals = time_intervals
        if ticks_per_interval is not None or time_intervals is not None:
            if self.active_network.network_type == 'continuous':
                self.active_network.max_ticks = self.active_network.ticks_per_interval * self.active_network.time_intervals + 1
            else:
                self.active_network.max_ticks = self.active_network.ticks_per_interval * self.active_network.time_intervals
        print("ticks_per_interval = ", self.active_network.ticks_per_interval)
        print("time_intervals = ", self.active_network.time_intervals)
        print("max_ticks = ", self.active_network.max_ticks)
        
    def add_net(self, net):
        if net not in self.networks:
            net.simulator = self
            self.networks.append(net)

        if (self.active_network == None):
            self.active_network = net

    def delete_net(self, net):
        self.networks.remove(net)

        if (self.active_network == net):
            self.active_network = self.networks[0]

    def delete_all_nets(self):
        self.networks = []

    # @dispatch(Network)
    def use_gui(self, net):
        net.simulator = self
        net.visualized = True
        program = main_viewer_tk(input_net=net)
        self.gui_program = program
        program.window.mainloop()
    
    def view_units(self, net, cell_size=9, cell_spacing=3, layout=None, plotCol=None, auto_plot=False):
        net.simulator = self
        net.visualized = True

        # If forcing auto plot, wipe any custom plotRow layout/state
        if auto_plot and layout is None:
            # simplest: remove both so viewer behaves like "no layout exists"
            if hasattr(net, "plot_layout"):
                net.plot_layout = None
            if hasattr(net, "_plotrow_state"):
                net._plotrow_state = None

        program = main_viewer_tk(input_net=net)
        self.gui_program = program
        program.start_unit_viewer()

        if plotCol is not None:
            net.plotCol = int(plotCol)

        if layout is not None:
            net.plot_layout = layout

        program.unit_viewer.window.mainloop()

    def view_units_plotrow(self, net, *plotrow_cmds, cell_size=9, cell_spacing=3, reset=True, plotCol=None):
        """Convenience: build a layout from plotRow commands, then open unit viewer.

        Accepts either:
        - varargs: view_units_plotrow(net, "plotRow ...", "plotRow ...", ...)
        - a single list/tuple: view_units_plotrow(net, ["plotRow ...", "plotRow ...", ...])
        """
        # If user passed one list/tuple, treat it as the sequence of commands
        if len(plotrow_cmds) == 1 and isinstance(plotrow_cmds[0], (list, tuple)):
            cmds = list(plotrow_cmds[0])
        else:
            cmds = list(plotrow_cmds)

        layout, state = build_layout(net, *cmds, reset=reset, plotCol=plotCol)
        self.view_units(net, cell_size=cell_size, cell_spacing=cell_spacing, layout=layout, plotCol=plotCol)
        
    def view_links(self, net, cell_size=0, cell_spacing=0):
        net.simulator = self
        net.visualized = True
        program = main_viewer_tk(input_net=net)
        self.gui_program = program
        program.start_link_viewer(cell_size=cell_size, cell_spacing=cell_spacing)
        program.link_viewer_gui.window.mainloop()

    def train_network_with_gui(self, net, epochs, batch_size,report_interval=network_params.PAR_N_reportInterval):
        """
        :param net: input network
        :param epochs: the number of weight updates to make
        :param batch_size: the number of examples to run before a weight update (if set to 0, the entire example set is run)
        :param report_interval:
        :return:
        initialize a new gui program for the network <net>, also create a thread for training the network with
        the given parameters. The training thread starts after gui program starts, and will end once the gui ends.
        """
        assert self.gui_program is None     # since we are initializing a new gui program
        net.simulator = self
        net.visualized = True
        q = Queue()
        # stop_event.wait() will be blocked until stop_event be true.
        # immediately after entering dispatch(), stop_event <- False
        # immediately after update_canvas (or confirm no update_canvas is necessary) in gui, stop_event <- True
        # therefore during dispatch(), training thread is blocked, and it is unblocked either when gui done checking or
        # updating, or time limit is reached.
        stop_event = threading.Event()
        stop_event.set()
        threads = []
        t2 = Thread(target=self.training_thread, args=(net, q, epochs, batch_size, report_interval, stop_event))
        t2.setDaemon(True)  # training will terminate once the gui ends
        threads.append(t2)
        t2.start()
        program = FrameExamplesProgram(input_net=net, stop_event=stop_event)
        self.gui_program = program
        q.put(True)
        program.window.mainloop()

    def train_network_with_main(self, net, epochs, batch_size,report_interval=network_params.PAR_N_reportInterval):
        """
        :param net: input network
        :param epochs: the number of weight updates to make
        :param batch_size: the number of examples to run before a weight update (if set to 0, the entire example set is run)
        :param report_interval:
        :return:
        initialize a new gui program for the network <net>, also create a thread for training the network with
        the given parameters. The training thread starts after gui program starts, and will end once the gui ends.
        """
        assert self.gui_program is None     # since we are initializing a new gui program
        net.visualized = True
        q = Queue()

        stop_event = threading.Event()
        stop_event.set()
        threads = []
        t2 = Thread(target=self.training_thread, args=(net, q, epochs, batch_size, report_interval,stop_event))
        t2.setDaemon(True)  # training will terminate once the gui ends
        threads.append(t2)
        t2.start()
        program = main_viewer_tk(input_net=net, stop_event=stop_event)
        self.gui_program = program
        q.put(True)
        program.window.mainloop()

    def view(self):
        pass

    def training_thread(self, net,signal_queue, epochs, batch_size,
                        report_interval=network_params.PAR_N_reportInterval, stop_event=None):
        while signal_queue.empty():
            pass
        net.train(epochs, batch_size, report_interval=report_interval, stop_event=stop_event)
        t1 = Thread(target=net.plotting_thread, daemon=True, args=[[]])
        t1.start()

    def generate_constant_output_history(self, net):
        temp_program = FrameExamplesProgram(input_net=net)
        if self.saved_history is None:
            temp_program.run_example_wrapper(example_to_run=0)
            self.saved_history = temp_program.curr_ex_history
        temp_program.window.destroy()
        del temp_program
        return self.saved_history


    def update_display_caller(self, last_ex_trained, s):
        self.gui_program.check_for_update(last_ex_trained, s)
    
    def graph_viewer_warn(self, message):
        self.gui_program.show_warning_message(message)
    
    def training_stop_complete(self):
        self.gui_program.update_stop_radio()

def runtk():
    # test for xor
    s_xor = Simulator("simulator for xor")
    mynet = Network("mynet")  # default time intervals (1) ->
    mynet.add_group("first", 2, "input", [], [])
    mynet.add_group("second", 2, None, ["dot"], ["sigmoid"])
    mynet.add_group("third", 1, "output", ["dot"], ["sigmoid"])
    mynet.connect_groups("first", "second", "uniform")
    mynet.connect_groups("second", "third", "uniform")
    mynet.load_example_set("XOR", False, "examples/example_files/xor_sparse.ex.txt")


# if __name__ == "__main__":
#     # test for xor
#
#     s_xor = Simulator("simulator for xor")
#     mynet = Network("mynet")  # default time intervals (1) ->
#     mynet.add_group("first", 2, "input", [], [])
#     mynet.add_group("second", 2, None, ["dot"], ["sigmoid"])
#     mynet.add_group("third", 1, "output", ["dot"], ["sigmoid"])
#     mynet.connect_groups("first", "second", "uniform")
#     mynet.connect_groups("second", "third", "uniform")
#     mynet.load_example_set("XOR",  False, "examples/example_files/xor_sparse.ex.txt")
#     s_xor.use_gui(mynet)

    # test for the 3000 nodes example

    # s_large = Simulator("simulator for 3000 nodes")
    # training_net = s_large.create_net(name="train", learning_rate=0.2)
    # training_net.add_group(name="first", num_units=182, group_type="input", input_transforms=[], output_transforms=[])
    # training_net.add_group(name="second", num_units=122, group_type="hidden", input_transforms=["dot"],
    #                        output_transforms=["sigmoid"])
    # training_net.add_group(name="third", num_units=3698, group_type="output", input_transforms=["dot"],
    #                        output_transforms=["sigmoid"])
    # training_net.connect_groups(outgoing_group="first", incoming_group="second", link_type="uniform")
    # training_net.connect_groups(outgoing_group="second", incoming_group="third", link_type="uniform")
    # training_net.load_example_set(
    #     example_set_name="TRAINING_EX", proc=False,
    #     file_name="exp_english_distribution=uniform_frequencyMinimum=1_dropType=linear_distributionType=probabilistic_dropAmount=0_shuffle=True.ex.txt",
    #     num_examples_loaded=1)
    #
    # with open('gui/history.pkl', 'rb') as input:
    #     history = pickle.load(input)
    #     print(history)
    # input.close()
    # s_large.use_gui(training_net, history)

# if __name__ == "__main__":
#     thd=threading.Thread(target=runtk)
#     thd.daemon = True  # background thread will exit if main thread exits
#     thd.start()  # start tk loop
#
#     while True:  # run in main thread
#         x = input("Enter a value or Q to quit: ")
#         if x.lower() == 'q':
#             exit()
#         print('Terminal entered', x)




