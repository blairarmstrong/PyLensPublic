import re
from typing import Optional

# EXAMPLE FIELDS
DEF_E_FREQUENCY = 1.0


class Example:
    """It stores information related to one example of the
    ExampleSet and has a list of events which will be executed by the neural network.
    """

    name = None  #: str
    num = 0  # : int
    num_events = 0  #: int
    event = []  #: List[Event]
    curr_ev_index = 0
    set = None  #: ExampleSet
    next = None  #: Example
    frequency: float
    probability = 0.0  #: float
    events_data = []
    event_headers = []
    pre_proc_name = None
    post_proc_name = None
    total_max_time = 0.0
    total_min_time = 0.0
    example_train_error = []
    example_test_error = 0.0

    def __init__(self, example_set, frequency=DEF_E_FREQUENCY):
        self.frequency = frequency
        self.set = example_set
        self.event = []
        self.events_data = []
        self.event_headers = []
        self.network = self.set.network
        self.example_train_error = []
        self.example_test_error = 0.0
        self.plot = False

    def pre_proc(self):
        """initialize and execute the code associated with this example's proc"""
        if self.pre_proc_name in self.set.procs_dict:
            proc_code = self.set.procs_dict[self.pre_proc_name]
            exec(proc_code)
        else:
            exec("print('error: no pre example proc defined for this event')")

    def post_proc(self):
        """initialize and execute the code associated with this example's proc"""
        if self.post_proc_name in self.set.procs_dict:
            proc_code = self.set.procs_dict[self.post_proc_name]
            exec(proc_code)
        else:
            exec("print('error: no post example proc defined for this event')")

    def get_total_time(self):
        """Calculate the total time for max and min time"""
        for event in self.event:
            if event.max_time is not None:
                self.total_max_time += event.max_time
            if event.min_time is not None:
                self.total_min_time += event.min_time

    def iterate_event(self):
        """ Return the event at curr_ev_index and increment curr_ev_index by 1
        """
        if self.curr_ev_index < self.num_events:
            event = self.event[self.curr_ev_index]
            self.curr_ev_index += 1
            return event
        print("Error: already reached end of events list!")
        return None

    def current_event(self):
        """Return the current event in the event_list based on the iterator"""
        if self.event:
            return self.event[self.curr_ev_index]
        print("Error: no event exists here")
        return None

    def next_event(self):
        """ Return the next event after index curr_ev_index if it exists; else return none
        """
        if self.curr_ev_index + 1 < self.num_events:
            return self.event[self.curr_ev_index + 1]
        print("Error: no event exists here")
        return None

    def first_event(self):
        """Return the first event of the iterator"""
        if self.event:
            return self.event[0]
        print("Error: no event exists here")
        return None

    def last_event(self):
        """Return the last event of the iterator"""
        if self.event:
            return self.event[-1]
        print("Error: no event exists here")
        return None

    def check_num_events(self, example_name):
        if "{" in example_name and "}" in example_name:
            regex = re.compile("( )[0-9]+(\\n|$)")
            matched = regex.search(example_name)
            if matched is not None:
                start = matched.start()
                end = matched.end()
                num = example_name[start: end].strip()
                if num.isdigit():
                    self.num_events = int(num)
                    example_name = example_name.replace(example_name[start: end], '')
        return example_name

    def parse_example_arguments(self, example_array: str) -> Optional[str]:
        """ Parse through example_array to find Example arguments and set the values
        in Example accordingly

        :param example_array:
        :type example_array:
        :return: new example array after the parsed parameters are removed
        """
        example_array += "\n"
        if "name:" in example_array:
            index = example_array.find("name:")
            find_newline = example_array[index:].find("\n") + index
            example_name = example_array[index + len("name:"): find_newline].strip()
            example_name = self.check_num_events(example_name)
            if example_name == "":
                self.name = self.set.example.index(self)
            else:
                self.name = example_name
                example_array = example_array.replace(example_array[index: find_newline + 1], '')

        if "freq:" in example_array:
            index = example_array.find("freq:")
            find_newline = example_array[index:].find("\n") + index
            example_freq = example_array[index + len("freq:"): find_newline].strip()
            reg = re.compile(r"([0-9]+\.[0-9]+)|[0-9]+")
            if reg.match(example_freq):
                self.frequency = float(example_freq)
                example_array = example_array.replace(example_array[index: find_newline + 1], '')
            else:
                self.set.parse_error("missing value after \"freq:\" in header of example " +
                                     str(self.example.set.example.index(self.example)))
                return None

        for proc in ["pre_proc:", "post_proc:"]:
            if proc in example_array:
                if "[" in example_array:
                    brac_index = example_array.find("[")
                    index = example_array.find(proc)
                    if brac_index > index:
                        find_newline = example_array[index:].find("\n") + index
                        example_proc = example_array[index + len(proc): find_newline].strip()
                        if proc == "pre_proc:":
                            self.pre_proc_name = example_proc
                        else:
                            self.post_proc_name = example_proc
                        example_array = example_array.replace(example_array[index: find_newline + 1], '')
                else:
                    index = example_array.find(proc)
                    find_newline = example_array[index:].find("\n") + index
                    example_proc = example_array[index + len(proc): find_newline].strip()
                    if proc == "pre_proc:":
                        self.pre_proc_name = example_proc
                    else:
                        self.post_proc_name = example_proc
                    example_array = example_array.replace(example_array[index: find_newline + 1], '')

        regex = re.compile("(^|\\n)[0-9]+(\\n|$)")
        matched = regex.search(example_array)
        if matched is not None:
            start = matched.start()
            end = matched.end()
            num = example_array[start: end].strip()
            if num.isdigit():
                self.num_events = int(num)
                example_array = example_array[:start] + example_array[end:]

        return example_array

    def parse_example_string(self, example_array: str):
        """ Sorts the string of an Example, example_array, into event headers and event dada
        :param example_array: The string of an Example
        :type example_array: str
        :return:
        """
        # print("event header", example_array)
        example_array = example_array.strip()
        if example_array[0].isdigit():
            num_event_str = ""
            i = 0
            while example_array[i].isdigit():
                num_event_str += example_array[i]
                i += 1

            self.num_events = int(num_event_str.strip())
            self.events_data = re.split(r'\[.+\]', example_array)
        else:
            if not self.num_events:
                self.event_headers = re.findall(r'\[(.+)\]', example_array)
                if len(self.event_headers) == 0:
                    self.num_events = 1
                else:
                    self.num_events = len(self.event_headers)
                    self.events_data = re.split(r'\[.+\]', example_array)
            else:
                self.event_headers = re.findall(r'\[(.+)\]', example_array)
                if len(self.event_headers) == 0:
                    self.events_data = example_array.split("\n")
                else:
                    self.num_events = len(self.event_headers)
                    self.events_data = re.split(r'\[.+\]', example_array)

    def write_example(self) -> str:
        """Write the example to the file"""
        s_output = ""
        s_output += self.write_example_header()
        for event in self.event:
            s_output += event.write_event()
        s_output += ";\n"
        return s_output

    def write_example_header(self) -> str:
        """Write the example header field values to the file"""
        output = ""
        if self.pre_proc_name is None:
            output += "pre_proc: " + "-" + "\n"
        else:
            output += "pre_proc: " + self.pre_proc_name + "\n"
        if self.post_proc_name is None:
            output += "post_proc: " + "-" + "\n"
        else:
            output += "post_proc: " + self.post_proc_name + "\n"
        output += "name: " + str(self.name) + "\n"
        output += "freq: " + str(self.frequency) + "\n"
        output += str(len(self.event)) + "\n"
        output += "\n"
        return output

    def print_out(self):
        """Print out the example"""
        self.print_out_example()

    def print_out_example(self, printing=True, tabs=0) -> str:
        """ Prints out the instance variables of an Example and of its Events
        This function is a work in progress for testing purposes.
        Each new layer of composition is indicated by indent.

        If calling this function directly, please leave
        printing=True and tabs=0 to their default values.
        """
        example = self
        lst = [("Obj", "Example"), ("name", example.name), ("num", example.num)]
        if example.next is not None:
            next_name = example.next.name
        else:
            next_name = None
        str_output = ""
        lst.extend(
            [("numEvents", example.num_events), ("pre_proc", example.pre_proc_name),
             ("post_proc", example.post_proc_name), ("set.name", example.set.name),
             ("next.name", next_name), ("frequency", example.frequency), ("probability", example.probability)])
        str_output += format_object_line(lst, tabs)
        for event_num in range(len(example.event)):
            event = example.event[event_num]
            str_output += event.print_out_event(False, tabs + 1)
        if printing:
            print(str_output)
        return str_output


def tab(char=1) -> str:
    """Add tab space to the print output"""
    return "     " * char


def format_object_line(lst, num_tabs=0) -> str:
    """format the print output line"""
    str_output = tab(num_tabs)
    for item in lst:
        str_output += item[0] + " = " + str(item[1]) + ", "
    return str_output + "\n"
