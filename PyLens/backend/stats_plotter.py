# import pyformulas as pf
from matplotlib.animation import FuncAnimation
from .parameters import NetworkParameters
import copy
import numpy as np
import matplotlib.pyplot as plt
from time import time
from humanfriendly import format_timespan
from queue import Queue
import matplotlib
# matplotlib.use('TkAgg')
# plt.switch_backend('agg')
network_params = NetworkParameters()


class StatsPlotter:
    """
    Handles the accumulation of training statistics and manages real-time plotting.

    This class records key training statistics, generates real-time plots, and prints progress reports
    to the console.

    Attributes:
        network (object): The neural network being monitored.
        points_on_plot (int): Number of most recent points shown on the graph. If 0, plots all points.
        stats_plotted (list): List of statistics to be plotted.
        plot_colors (list): List of colors for different runs in the plots.
        report_interval (int): Number of epochs between each printed report.
        live_update (bool): If True, updates the plots in real-time.
        print_reports (bool): If True, prints training progress reports.
        run_number (int): Counts the number of runs. A new run is counted when training is paused and resumed.
        runs (list): List of tuples `(run_number, last_epoch_of_this_run)`.
        report_stats (dict): Stores intermediate statistics for reporting training progress.
        progress_stats (dict): Stores accumulated statistics for tracking progress.
    """

    def __init__(self, network, points_on_plot=0,
                 stats_plotted=None, plot_colors=None,
                 report_interval=1,
                 ):
        self.first = True
        self.network = network
        self.points_on_plot = points_on_plot
        if stats_plotted is None:
            stats_plotted = ["error"]
        self.stats_plotted = stats_plotted
        if plot_colors is None:
            plot_colors = ["blue", "orange", "green", "red",
                           "purple", "brown", "pink" "gray", "olive", "cyan"]
        self.plot_colors = plot_colors
        self.report_interval = report_interval
        self.live_update = True
        self.print_reports = True
        self.run_number = 0
        self.runs = []
        self.live_update_closed = False
        self.fig, self.ax, self.screen = None, None, None
        self.plot_variable = "error"

        self.previous_length = -1

        self.group_to_plot = -1
        self.unit_to_plot = 0


        # These are initialized when training begins and are used for progress reports
        self.report_stats = {
            "last_report_update_no": 0,
            "last_report_time": 0,
            "time_per_update": 0.0,
            "training_start_time": 0.0,
            "lwd_times_derivs": 0.0,  # sum of last_weight_delta * deriv
            "squared_lwd": 0.0,  # sum of last_weight_delta ** 2
            "squared_derivs": 0.0,  # sum of deriv ** 2
            "squared_weights": 0.0  # sum of weight ** 2
        }
        # stats for training progress
        self.progress_stats = {
            "update_no": [],
            "error": [],
            "unit_cost": [],
            "weight_cost": [],
            "grad_lin": [], # TypeError
            "time_used": [],
            "remaining_time": [],
            "unit_output": [],
            "unit_input_deriv": [],
            "unit_output_deriv": [],
        }
        self.last_progress_stats = {
            "update_no": [0],
            "error": [0],
            "unit_cost": [0],
            "weight_cost": [0],
            "grad_lin": [0],
            "time_used": [0],
            "remaining_time": [0],
            "unit_output": [0],
            "unit_input_deriv": [0],
            "unit_output_deriv": [0],
        }
    def set_training_start_time(self, start_time):
        """
        Set training start time.
        """
        self.report_stats["training_start_time"] = start_time
        
    def toggle_live_update(self, live_update=True):
        """
        Turn live update on or off.
        """
        self.live_update = live_update

    def toggle_print_reports(self, print_reports=True):
        """
        Turn print reports on or off.
        """
        self.print_reports = print_reports

    def add_plotting_stats(self, stats_plotted=None):
        """
        Add a stat to be included in the live plots.
        """
        if stats_plotted is None:
            self.stats_plotted = ["error"]
        elif type(stats_plotted) == str:
            self.stats_plotted.append(stats_plotted)
        else:
            self.stats_plotted.extend(stats_plotted)

    def set_plotting_stats(self, stats_plotted=None):
        """
        Set which stats are to be included in the live plots.
        """
        if stats_plotted is None or not isinstance(stats_plotted, list):
            self.stats_plotted = ["error"]
        else:
            self.stats_plotted = stats_plotted

    def set_plot_colors(self, plot_colors):
        """
        Sets the colors used in plots.

        Args:
            plot_colors (list): List of colors to be used in order of runs.
        """
        if not isinstance(plot_colors, list):
            print("error: set plot colors should input a list")
        else:
            self.plot_colors = plot_colors

    def count_run(self, epoch_number):
        """
        Record the current pause as a "run" for the plotting.
        """
        self.runs.append(
            (self.run_number, epoch_number // self.report_interval))
        self.run_number += 1

    def plot_stats(self, y_axis_value, update_no_list, fig, axis, color, com_graph=False):
        """
        Saves accumulated training statistics to a file.

        Args:
            file_path (str): Path to the file where stats should be saved.
        """
        y_values = []

        def make_plot(axis, update_no_list, y_values, points_on_plot, color, com_graph=False):
            if not com_graph:
                if len(update_no_list) > points_on_plot and points_on_plot not in [0, 1]:
                    axis.set_xlim(
                        update_no_list[-self.points_on_plot], update_no_list[-1])
            else:
                axis.set_xlim(0, update_no_list[-1])

            previous_run_end = 0
            for run in self.runs:
                run_number, x = run
                axis.plot(update_no_list[previous_run_end:x],
                          y_values[previous_run_end:x], color=self.plot_colors[run_number])
                previous_run_end = x
            axis.plot(update_no_list[previous_run_end:],
                      y_values[previous_run_end:], color=self.plot_colors[self.run_number])

            # axis.set_xlabel("epochs")
            axis.set_ylabel(live_plot_y_axis)
            fig.canvas.draw()

        live_plot_y_axis = y_axis_value.lower()
        if live_plot_y_axis in self.progress_stats:
            y_values = self.progress_stats[live_plot_y_axis]
        else:
            y_values.append(0)
        make_plot(axis, update_no_list, y_values,
                  self.points_on_plot, color, com_graph)

    def update_plots(self, update_no_list, fig, axes):
        """
        Updates the live training plots.

        Args:
            update_no_list (list): List of x-axis values representing update numbers.
            fig (matplotlib.figure.Figure): The figure object for the plot.
            axes (matplotlib.axes.Axes or list): The axes object(s) where plots will be updated.
        """
        if len(self.stats_plotted) > 1:
            for l in range(len(self.stats_plotted)):
                self.plot_stats(self.stats_plotted[l], update_no_list, fig, axes[l],
                                self.plot_colors[self.run_number])
        else:
            self.plot_stats(
                self.stats_plotted[0], update_no_list, fig, axes, "red")

    def plot_example_error_progress(self, example, axis, example_num, epochs):
        """
        Plots the error progression over epochs for a given example.

        Args:
            example (object): The example whose error progression is being plotted.
            axis (matplotlib.axes.Axes): The axis to plot on.
            example_num (int): The index number of the example.
            epochs (int): Number of epochs to be plotted.
        """
        epoch_list = [i + 1 for i in range(epochs)]
        plot_title = self.network.name + " example set's example " + str(
            example_num) + " error vs epochs using " + self.network.update_method
        axis.set_title("example error vs epoch")
        axis.plot(epoch_list, example.example_train_error, label=plot_title)
        axis.set_xlabel("epochs")
        axis.set_ylabel("example error")
        axis.legend()
        plt.show()

    def _prepare_for_report(self):
        """
        Prepares for reporting progress by printing the report header and initializing the report variables.
        """
        # Print report header
        if self.print_reports:
            print("".join(f"{h:^14}" for h in ["Update", "Error", "UnitCost", "Wgt.Cost", "Grad.Lin", "TimeUsed", "TimeLeft"]))
        # initialize variables used for report progress
        self.report_stats["last_report_update_no"] = 0
        self.report_stats["last_report_time"] = 0
        self.report_stats["time_per_update"] = 0.0
        self.report_stats["training_start_time"] = 0.0
        self.report_stats["lwd_times_derivs"] = 0.0
        self.report_stats["squared_lwd"] = 0.0
        self.report_stats["squared_derivs"] = 0.0
        self.report_stats["squared_weights"] = 0.0
    
    def dynamic_precision(self, value, len=5):
        """
        Determines the decimal precision dynamically based on the value.
        
        Args:
            value (float): The numerical value whose precision needs to be adjusted.
            length (int, optional): The desired total length of the formatted number. Default is 5.

        Returns:
            int: The number of decimal places to use.
        """
        if value > 10000:   # if it is too large, return 0
            return 0
        else:
            index = str(value).find(".")
            if index == -1: index = len # if it is an integer
            return len - index
    
    def simplified_format_timespan(self, time: float):
        """
        Formats a time span into a compact and human-readable format.

        Converts a numerical time span (in seconds) into a simplified format, e.g.,
        `1h30m` instead of `1 hour and 30 minutes`.

        Args:
            time_value (float): Time duration in seconds.

        Returns:
            str: A human-friendly formatted time string.
        """
        formatted_time = format_timespan(time)
        mapping_table = {"hours": "h", "minutes": "m", "seconds": "s", 
                         "hour": "h", "minute": "m", "second": "s", "and": " "}
        formatted_time = formatted_time.replace(" ", "")
        for key in mapping_table.keys():
            formatted_time = formatted_time.replace(key, mapping_table[key])
        return formatted_time

    def _print_report_field(self, value, integer=False, last_field=False):
        """
        Prints a formatted report field value.

        Args:
            value (float or str): The value to print.
            integer (bool, optional): If True, the value is treated as an integer. Default is False.
            last_field (bool, optional): If True, prints a newline after the value. Default is False.
        """
        # end the print statement with a new line only if it is the last field
        if last_field:
            end = '\n'
        else:
            end = ''
        if value is None:
            print("{:^14}".format("-"), end=end)
        elif integer:
            print("{:^14}".format(value), end=end)
        else:
            if type(value) == str:
                print("{:^14}".format(value), end=end)
            else:
                value = float(value) # for pytorch
                print("{:^14.{}f}".format(value, self.dynamic_precision(value)), end=end)

    def report_progress(self, update_no, total_updates, first_update, plot_queue):
        """
        Reports the training progress made so far and estimates the remaining time.

        Args:
            update_no (int): Number of updates performed so far (i.e., batch number).
            total_updates (int): Total number of updates planned (i.e., number of epochs).
            first_update (bool): Whether this is the first update (affects gradient linearity calculation).
            plot_queue (Queue): A queue used for handling plot updates.

        """

        error = self.network.batch_error(self.network.batch_errors)
        unit_cost = self.network.sum_batch_unit_cost(self.network.batch_unit_costs)
        weight_cost = self.network.weight_cost()
        if update_no == 1:
            grad_lin = None  # because nfo from last update is needed
        else:
            grad_lin = self.network.gradient_linearity()
        time_used_raw = time() - self.report_stats["training_start_time"]
        remaining_time_raw = self.network.remaining_time(update_no, total_updates)
        time_used = self.simplified_format_timespan(time_used_raw)
        remaining_time = self.simplified_format_timespan(remaining_time_raw)
        update_no = update_no + self.network.num_update

        
        if self.print_reports:
            self._print_report_field(
                update_no, integer=True)
            self._print_report_field(error)
            self._print_report_field(unit_cost)
            self._print_report_field(weight_cost)
            self._print_report_field(grad_lin)
            self._print_report_field(time_used)
            self._print_report_field(remaining_time, last_field=True)

        # Reset network-level report stats
        self.report_stats["lwd_times_derivs"] = 0.0
        self.report_stats["squared_lwd"] = 0.0
        self.report_stats["squared_derivs"] = 0.0
        self.report_stats["squared_weights"] = 0.0

        self.progress_stats["update_no"].append(update_no)
        self.progress_stats["error"].append(error)
        self.progress_stats["unit_cost"].append(unit_cost)
        self.progress_stats["weight_cost"].append(weight_cost)
        self.progress_stats["grad_lin"].append(grad_lin)
        self.progress_stats["time_used"].append(time_used_raw)
        self.progress_stats["remaining_time"].append(remaining_time_raw)
        self.progress_stats["unit_output"].append(self.network.groups[self.group_to_plot].output_matrix[self.unit_to_plot])
        self.progress_stats["unit_input_deriv"].append(self.network.groups[self.group_to_plot].input_derivs[self.unit_to_plot])
        self.progress_stats["unit_output_deriv"].append(self.network.groups[self.group_to_plot].output_derivs[self.unit_to_plot])



        # plot_queue.append(first_update)

    def reset_stats(self):
        """
        Reset the accumulated stats variables to default.
        """
        # no need to copy if nothing has been accumulated
        if self.progress_stats['update_no'] != []:
            self.last_progress_stats = copy.deepcopy(self.progress_stats)
        self.report_stats["lwd_times_derivs"] = 0.0
        self.report_stats["squared_lwd"] = 0.0
        self.report_stats["squared_derivs"] = 0.0
        self.report_stats["squared_weights"] = 0.0

        self.progress_stats["update_no"] = []
        self.progress_stats["error"] = []
        self.progress_stats["unit_cost"] = []
        self.progress_stats["weight_cost"] = []
        # self.progress_stats["grad_lin"] = []
        self.progress_stats["time_used"] = []
        self.progress_stats["remaining_time"] = []

    def save_stats(self, file_path):
        """
        Saves the accumulated training statistics to a CSV file.

        Args:
            file_path (str): The file path where the statistics should be saved.
        """
        with open(file_path, 'w') as f:
            for key in self.progress_stats.keys():
                f.write("%s,%s\n" %
                        (key, self.progress_stats[key]) + str(self.runs))

    # comment because not used
    #def live_graphs(self, graph_title):
    #    # print(first, self.fig, self.ax, first, quantity_to_graph)

    #    if self.first:
    #        print("first")
    #        self.fig, self.ax = plt.subplots()

    #        self.screen = pf.screen(title=graph_title)
    #        self.first = False

    #    # curr_time = time()
    #    # if curr_time - self.clock_time < self.plot_update_time:
    #    #     return

    #    # self.clock_time = curr_time
    #    # if len(self.progress_stats["update_no"]) == self.previous_length:
    #    #     return
    #    # self.previous_length = len(self.progress_stats["update_no"])

    #    quantity_to_graph = self.plot_variable

    #    if quantity_to_graph == "error":
    #        # self.ax.plot(
    #        #     self.progress_stats["update_no"], self.network.progress_stats["error"], c='black')
    #        print("graphing error")
    #        self.ax.plot(self.network.progress_stats["error"], c='black')
    #        self.ax.set_title(graph_title)
    #        self.ax.set_xlabel("Reporting Interval")
    #        self.ax.set_ylabel("Error")
    #    elif quantity_to_graph == "unit_cost":
    #        self.ax.plot(self.progress_stats["update_no"],
    #                     self.progress_stats["unit_cost"], c='black')
    #        self.ax.set_title(graph_title)
    #        self.ax.set_xlabel("Reporting Interval")
    #        self.ax.set_ylabel("unit_cost")

    #    elif quantity_to_graph == "weight_cost":
    #        self.ax.plot(self.progress_stats["update_no"],
    #                     self.progress_stats["weight_cost"], c='black')
    #        self.ax.set_title(graph_title)
    #        self.ax.set_xlabel("Reporting Interval")
    #        self.ax.set_ylabel("weight_cost")

    #    elif quantity_to_graph == "grad_lin":
    #        self.ax.plot(self.progress_stats["update_no"],
    #                     self.progress_stats["grad_lin"], c='black')
    #        self.ax.set_title(graph_title)
    #        self.ax.set_xlabel("Reporting Interval")
    #        self.ax.set_ylabel("grad_lin")

    #    elif quantity_to_graph == "time_used":
    #        self.ax.plot(self.progress_stats["update_no"],
    #                     self.progress_stats["time_used"], c='black')
    #        self.ax.set_title(graph_title)
    #        self.ax.set_xlabel("Reporting Interval")
    #        self.ax.set_ylabel("time_used")

    #    elif quantity_to_graph == "remaining_time":
    #        self.ax.plot(self.progress_stats["update_no"],
    #                     self.progress_stats["remaining_time"], c='black')
    #        self.ax.set_title(graph_title)
    #        self.ax.set_xlabel("Reporting Interval")
    #        self.ax.set_ylabel("remaining_time")

    #    else:
    #        self.ax.plot(self.progress_stats["update_no"],
    #                     self.progress_stats[quantity_to_graph], c='black')
    #        self.ax.set_title(graph_title)
    #        self.ax.set_xlabel("Reporting Interval")
    #        self.ax.set_ylabel(quantity_to_graph)

    #    # If we haven't already shown or saved the plot, then we need to draw the figure first...
    #    self.fig.canvas.draw()

    #    image = np.fromstring(
    #        self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #    image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

    #    # if self.screen:
    #    self.screen.update(image)
    #    #
    #    # try:
    #    #     self.screen.update(image)
    #    # except Exception as e:
    #    #     print("plot closed")
    #    #     self.screen.close()
    #    #     self.network.plot_thread_created = False

    ## def close_live_graph(self):
    ##     plt.close(self.fig)
    ##     self.screen.close()
