import numpy as np
import tkinter as tk
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.animation as animation
import sys
from sympy import Symbol

sys.path.insert(0, ".")

class GraphViewer:
    """
    Graph viewer can plot a range of network properties during training.
    """

    def __init__(self, parent, network, plot_variable: Symbol, special_variables: dict={}, 
                 window_name: str="Graph Viewer", update_after: int=3):
        """
        Initializes a GraphViewer object to display a graph of a given variable over time.
        
        Args: 
            parent: the parent window of the graph viewer.
            network (Network): the network object that the graph viewer is associated with.
            plot_variable (Symbol): the variable to plot.
            special_variables (dict): a dictionary of special variables to plot.
            window_name (str): the name of the window.
            update_after (int): the update frequency of the graph.
        """
        self.network = network
        self.window_name = window_name
        self.plot_variable = plot_variable
        self.update_after = update_after
        self.interval = 100  # draw canvas interval in ms
        self.animation = None
        self.window = tk.Tk()
        self.window.title(self.window_name)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing) # Make sure the program exits when clicking the 'x'
        self.parent = self.window
        self.special_variables = special_variables
        
        # Set up figure
        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot(111)
        self._set_plot_labels()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # Matplotlib menu
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.window)
        self.toolbar.update()
        
        # Initial data values
        self.plot_data = [[]]  # list of lists of data to plot
        self.prev_trace_end = 0
        self.prev_trace_index = 0
        self.x_data = [[]]
        
        # Starting plot limit values
        self.x_max = 1
        self.x_min = 0
        self.y_max = .000000001
        self.y_min = 0
        self.graph_clock = 0
        self.update_xy_limits()
        self.min_x_data = float('inf')

        # First trace
        line = self._create_new_artist([], [])  # first trace in graph_viewer
        self.artists = [line]  # list of all traces

        # Button to open Trace Management window
        self.trace_management_button = tk.Button(
            self.window, text="Trace Management", command=self.open_trace_management_window
        )
        self.trace_management_button.pack(side=tk.LEFT, padx=10, pady=5)

    def update_clock(self, x):
        """
        Updates the graph clock by adding x to it."""
        self.graph_clock += x
    
    def set_clock(self, x):
        """
        Sets the graph clock to x.
        """
        self.graph_clock = x

    def update_x_data(self, x):
        """
        Updates the x_data list with the given x value.
        """
        if len(self.x_data) == 1:
            data_point = x + self.prev_trace_end
        else:
            data_point = x
        # first data point of the active trace
        if len(self.x_data[-1]) == 0:
            if data_point < self.min_x_data:    # update min_x_data
                self.min_x_data = data_point

        self.x_data[-1].append(data_point)

    def update_plot(self, frame=None):
        """
        Updates the plot with the latest data.
        
        Args:
            frame: the frame to update.
        """
        dimensions_match = True
        plot_length = len(self.network.stats_plotter.progress_stats["error"])
        if plot_length != len(self.network.stats_plotter.progress_stats["error"]):
            dimensions_match = False
        if dimensions_match:
            # update active trace (the last one in self.artists)
            self.artists[-1].set_xdata(self.x_data[-1])
            self.artists[-1].set_ydata(self.plot_data[-1])
            self.canvas.draw()

            self.fig.gca().relim()
            self.fig.gca().autoscale_view(True, True, True)
            
            return self.artists

    def update_trace(self):
        """
        Updates trace index and records previous trace end data to start a new trace.
        """
        self.prev_trace_end = self.network.stats_plotter.progress_stats["update_no"][-1]
        self.prev_trace_index = len(self.x_data)

    def restart(self):
        """
        Restarts plots from scratch by appending a new list (empty) for plotting to data.
        """
        self.plot_data.append([])
        self.x_data.append([])
        self.graph_clock = 0
        # Add random color for new trace
        new_color = (np.random.random(), np.random.random(), np.random.random())
        self.artists[-1].set_color(new_color)
        # Create new active trace
        self.artists.append(self._create_new_artist([], []))

        if hasattr(self, 'trace_manager_window'):
            self.trace_manager_window.destroy()
            self.open_trace_management_window()

    def start(self):
        """
        Starts matplotlib's animation to continuously update the graph each self.interval ms.
        """
        if self.animation is None:
            self.animation = animation.FuncAnimation(
                self.fig, self.update_plot, interval=self.interval, repeat=False, blit=True, cache_frame_data=False
            )
        else:
            self.animation.event_source.start()
    
    def pause(self):
        """
        Pauses matplotlib's animation.
        """
        if self.animation is None:
            print("Warning, no animation to pause.")
        else:
            self.animation.event_source.stop()
    
    def _set_plot_labels(self):
        """
        Sets labels for ax if it exists.
        """
        update_frequency = {
            0: "example",
            1: "weight update",
            2: "completion of a batch",
            3: "epoch",
            4: "training and testing"
        }
        if self.ax:
            self.ax.set_xlabel(update_frequency[self.update_after])
            self.ax.set_ylabel(str(self.window_name))
            self.ax.set_title(str(self.window_name) + " over time")

    def _create_new_artist(self, x_data: list, y_data: list):
        """
        Returns a new line2D artist using x_data and y_data.

        Args:
            x_data: the x data to plot.
            y_data: the y data to plot
        """
        line, = self.ax.plot(x_data, y_data, color=(0, 0, 0))
        return line
    
    def set_xy_limits(self, x_min=0, x_max=2000, y_min=0, y_max=2000) -> None:
        """
        Sets the x and y limits of the graph.

        Args:
            x_min: the minimum x value.
            x_max: the maximum x value.
            y_min: the minimum y value.
            y_max: the maximum y value.
        """
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.canvas.draw()
    
    def update_xy_limits(self) -> None:
        """
        Updates the x and y limits of the graph to the current
        max and min values.
        """
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        
        try:
            self.canvas.draw()
        except: # for unit test only
            pass

    def on_closing(self):
        """
        Deletes the graph object from the network after closing.
        """
        self.window.destroy()
        self.network.graphs.remove(self)

    # ----------------------------------------------------------------
    #  Trace Management Window
    # ----------------------------------------------------------------

    def open_trace_management_window(self):
        """
        Opens a new window to manage existing traces:
          - Shows each trace's color
          - Shows whether it is the active (last) trace
          - Provides a button to clear (remove) that trace
        """
        self.trace_manager_window = tk.Toplevel(self.window)
        self.trace_manager_window.title("Trace Management")

        # Create header row
        header_frame = tk.Frame(self.trace_manager_window)
        tk.Label(header_frame, text="Trace").pack(side=tk.LEFT, padx=20)
        tk.Label(header_frame, text="Color").pack(side=tk.LEFT, padx=20)
        tk.Label(header_frame, text="Active?").pack(side=tk.LEFT, padx=20)
        header_frame.pack(pady=5, fill=tk.X)

        # Create a row for each trace
        for i, line in enumerate(self.artists):
            frame = tk.Frame(self.trace_manager_window)
            frame.pack(anchor="w", fill=tk.X, pady=2)

            # Trace label
            tk.Label(frame, text=f"Trace {i+1}").pack(side=tk.LEFT, padx=20)

            # Color
            color_value = line.get_color()
            if isinstance(color_value, tuple):
                # Convert floating (0 to 1) to #RRGGBB
                hex_color = "#{:02x}{:02x}{:02x}".format(
                    int(color_value[0]*255),
                    int(color_value[1]*255),
                    int(color_value[2]*255)
                )
            else:
                # If it’s already something valid for tkinter, keep it
                hex_color = color_value  

            # Show a small color swatch
            swatch_label = tk.Label(frame, bg=hex_color, width=4, height=1)
            swatch_label.pack(side=tk.LEFT, padx=20)

            # Active trace check
            # Here "active" is simply if it's the last in the list
            is_active = (i == len(self.artists) - 1)
            state_label = tk.Label(frame, text=("Yes" if is_active else "No"))
            state_label.pack(side=tk.LEFT, padx=20)

            # Clear button
            
            btn_clear = tk.Button(
                frame, text="Clear",
                command=lambda idx=i: self.clear_trace(idx), 
                state=tk.DISABLED if is_active else tk.NORMAL
            )
            btn_clear.pack(side=tk.LEFT, padx=20)

    def clear_trace(self, index):
        """
        Removes the given trace from the figure and from internal data structures.
        """
        if index < 0 or index >= len(self.artists):
            return  # Invalid index, do nothing

        # Remove the artist visually
        line_to_remove = self.artists[index]
        line_to_remove.remove()
        
        # Pop from the internal lists
        self.artists.pop(index)
        self.x_data.pop(index)
        self.plot_data.pop(index)

        # If the user removed the last (active) trace, you might want
        # to ensure there's still at least one trace, or re-create a new one.
        # Here’s a simple fallback:
        if len(self.artists) == 0:
            # Create a new one so the viewer doesn't break
            self.plot_data.append([])
            self.x_data.append([])
            self.artists.append(self._create_new_artist([], []))

        self.canvas.draw()

        # Also remove the row in the Trace Management window by closing + reopening
        # (Simple approach—if you want more elegant row-by-row removal, you'll have to
        #  keep references to the row widgets and destroy them individually.)
        if hasattr(self, 'trace_manager_window'):
            self.trace_manager_window.destroy()
            self.open_trace_management_window()
