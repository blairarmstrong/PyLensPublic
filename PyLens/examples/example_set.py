from typing import List, Optional
import re
import random
from .example import Example
from .event import Event
from .example_iterator import ExampleIterator
import codecs

# EXAMPLE SET FIELDS
DEF_S_PIPELOOP = True
DEF_S_MAXTIME = 1.0
DEF_S_MINTIME = 0.0
DEF_S_GRACETIME = 0.0
DEF_S_DEFAULTINPUT = 0.0
DEF_S_ACTIVEINPUT = 1.0
DEF_S_DEFAULTTARGET = 0.0
DEF_S_ACTIVETARGET = 1.0


class ExampleSet:
    """
    It stores a set of examples with similar properties
    on which the neural network will be trained or tested. Each ``.ex`` file contains information used to construct one ExampleSet.

    """
    name: str
    num_examples: int

    example = []  #: List[Example]  list of examples
    example_iterator: ExampleIterator
    example_index = []
    first_example = None  #: Example
    last_example = None  #: Example
    max_time: float
    min_time: float
    grace_time: float
    default_input: float
    active_input: float
    default_target: float
    active_target: float
    file_name: str
    pre_epoch_proc_name = None
    post_epoch_proc_name = None  # a code which needs to be implemented before loading in values
    post_update_proc_name = None
    input_group = None  # List[Group]
    target_group = None  # List[Group]
    num_loaded: int
    loading_order: str

    def __init__(self,
                 network,
                 proc_exists: bool,
                 name: str,
                 file_name: str,
                 input_groups,
                 target_groups,
                 default_input: int,
                 active_input: int,
                 default_target: int,
                 active_target: int,
                 loading_order="ORDERED",
                 def_s_pipe_loop=DEF_S_PIPELOOP,
                 def_s_max_time=DEF_S_MAXTIME,
                 def_s_min_time=DEF_S_MINTIME,
                 def_s_grace_time=DEF_S_GRACETIME,
                 mode="ORDERED",
                 num_loaded=None):

        self.name = name
        self.pipe_loop = def_s_pipe_loop
        self.max_time = def_s_max_time
        self.min_time = def_s_min_time
        self.grace_time = def_s_grace_time
        self.default_input = default_input
        self.active_input = active_input
        self.default_target = default_target
        self.active_target = active_target

        self.group_name = []
        self.num_examples = 0
        self.num_events = 0
        self.example = []
        self.file_name = file_name
        self.procs_dict = {}
        self.pre_epoch_proc_name = None
        self.post_epoch_proc_name = None
        self.post_update_proc_name = None
        if proc_exists:
            self.procs_file_name = self.file_name[0:-3] + "_procs.ex"
            procs_file = codecs.open(self.procs_file_name, 'r', 'utf-8')
            procs_file_str = ignore_commented_lines(procs_file.read())
            if not self.parse_procs(procs_file_str):
                print("Failed to parse procs in example set")
                return

        self.input_group = input_groups
        self.target_group = target_groups
        self.sort_mode = mode
        self.sparse_mode = False
        self.num_loaded = num_loaded
        self.loading_order = loading_order
        self.network = network

    @classmethod
    def initialize_example_set(cls, network,
                               proc_exists: bool,
                               name: str,
                               file_name: str,
                               input_groups,
                               target_groups,
                               default_input: int,
                               active_input: int,
                               default_target: int,
                               active_target: int,
                               loading_order="ORDERED",
                               def_s_pipe_loop=DEF_S_PIPELOOP,
                               def_s_max_time=DEF_S_MAXTIME,
                               def_s_min_time=DEF_S_MINTIME,
                               def_s_grace_time=DEF_S_GRACETIME,
                               mode="ORDERED",
                               num_loaded=None) -> Optional['ExampleSet']:
        """
        Initialize an ExampleSet object and call read_in_file to start loading in the examples from text example file
        """
        example_set = cls(network, proc_exists, name, file_name, input_groups, target_groups, default_input,
                          active_input, default_target, active_target, loading_order, def_s_pipe_loop, def_s_max_time,
                          def_s_min_time, def_s_grace_time, mode, num_loaded)

        if example_set.read_in_file() is False:
            print("Error loading in the values while reading the file")
            return None
        return example_set

    def pre_epoch_proc(self):
        """Initialize and execute the code associated with this example_set's pre epoch proc"""
        if self.pre_epoch_proc_name in self.procs_dict:
            proc_code = self.procs_dict[self.pre_epoch_proc_name]
            exec(proc_code)
        else:
            exec("print('error: no pre epoch proc defined for this event')")

    def post_epoch_proc(self):
        """Initialize and execute the code associated with this example_set's proc"""
        if self.post_epoch_proc_name in self.procs_dict:
            proc_code = self.procs_dict[self.post_epoch_proc_name]
            exec(proc_code)
        else:
            exec("print('error: no post epoch proc defined for this event')")

    def post_update_proc(self):
        """Initialize and execute the code associated with this example_set's post_update proc"""
        if self.post_update_proc_name in self.procs_dict:
            proc_code = self.procs_dict[self.post_update_proc_name]
            exec(proc_code)
        else:
            exec("print('error: no post epoch post_update proc defined for this event')")

    def iterate_example(self):
        """Iterate to the next example in the ordered example array"""
        return self.example_iterator.iterate_example()

    def set_sort_mode(self, sort_mode: str):
        """ Manually change sort mode to sort_mode
        :param sort_mode: new sort mode
        :type sort_mode: str
        """
        self.sort_mode = sort_mode
        self.example_iterator.reset_example_list()

    def get_first_example(self):
        """ Returns first example
        """
        return self.example_iterator.first_example()

    def get_last_example(self) -> Example:
        """ Returns last example
        """
        return self.example_iterator.last_example()

    def get_current_example(self) -> Example:
        """ Returns current example
        """
        return self.example_iterator.current_example()

    def get_prev_example(self) -> Example:
        """Returns the previous example"""
        return self.example_iterator.prev_example()

    def get_next_example(self) -> Example:
        """ Returns next example
        """
        return self.example_iterator.next_example()

    def read_in_file(self) -> bool:
        """ Return a list of strings separated by ";" from name .ex file and then
        fills S object with information from the file by calling read_example
        """
        # open file as string f
        example_set_file = codecs.open(self.file_name, 'r', 'utf-8')
        # split file by ";"
        file_str = ignore_commented_lines(example_set_file.read())
        self.is_sparse_format(file_str)
        split_list = file_str.split(";")
        example_list = []
        for example_string in split_list:
            example_list.append(example_string.strip())
        return self.read_example(example_list)

    def parse_procs(self, procs_file_str: str) -> bool:
        """ Parse through the procs and fills in self.procs_dict
        :return:
        """
        first_proc = procs_file_str.index("startofproc")
        reg = re.compile("[^(,\n)]+=[^(,\n)]+")
        global_var_lst = reg.findall(procs_file_str[0:first_proc])
        global_dict = {}
        for element in global_var_lst:
            key_value = element.split("=")
            global_dict[key_value[0].strip()] = key_value[1].strip()

        procs_file_str[:first_proc].split()
        procs_lst = procs_file_str.split("startofproc")
        procs_lst.pop(0)
        for proc in procs_lst:
            # error if } comes before {
            if "{" not in proc or "}" not in proc or proc.index("{") >= proc.index("}"):
                return self.parse_error("error in proc brackets ")

            proc_start_index = proc.index("{")
            proc_end_index = proc.index("}")
            name = proc[0:proc_start_index].strip()
            proc_str = proc[proc_start_index + 1:proc_end_index - 1].strip()
            reg2 = re.compile(r"def (\w+)\s*\((.*?)\):")
            func_lst = reg2.split(proc_str)
            proc_code = func_lst[-1].strip()
            parameters = func_lst[-2].split(",")
            for para in parameters:
                if para.strip() in global_dict:
                    proc_code = para.strip() + "=" + global_dict[para.strip()] + "\n" + proc_code
            self.procs_dict[name] = proc_code
        return True

    def is_sparse_format(self, example_set_string: str) -> bool:
        """ Returns true if the raw string text of example set file (after commented
        lines are ignored) looks like it's in sparse format
        :param example_set_string: raw string text of example set file
        :type example_set_string: str
        :rtype: bool
        """
        for value_type in ["i:", "t:", "b:"]:
            if value_type in example_set_string:
                self.sparse_mode = True
                return True
        self.sparse_mode = False
        return False

    def read_example(self, example_list: List[str]) -> bool:
        """
        Read the example_list from a ``.ex`` file, fill attributes of the example set,
        and register the example by calling ``register_example()``.

        Args:
            example_list (List[str]): Lines from the ``.ex`` file that represent a
                single example. Examples in the file are separated by semicolons (``;``).
        """
        example_list.pop()
        header_string = example_list[0]
        example_set_header = self.parse_example_set_header_string(header_string)
        if example_set_header is None:
            return False
        example_list[0] = example_set_header
        if example_list[0].strip() == '':
            example_list.pop(0)

        # if header is empty then remove header
        self.num_examples = len(example_list)

        # Create and load in examples
        for j in range(self.num_examples):
            example = Example(self)
            self.register_example(example)
            res = example.parse_example_arguments(example_list[j])
            if not res:
                return False
            else:
                example_list[j] = res
            example.parse_example_string(example_list[j].strip())
            if example.events_data != [] and example.events_data[0] == "":
                example.events_data.pop(0)

            for _ in range(example.num_events):
                new_event = Event(example)
                example.event.append(new_event)
            # Add and load in events to the example
            for i in range(example.num_events):
                if example.event_headers:
                    if example.event[i].parse_event_header_string(example.event_headers[i]) is False:
                        return False
                    if example.event[i].parse_event_list(example.events_data[i], self.sparse_mode) is False:
                        return False
                elif len(example.events_data) > 1:
                    if len(example.events_data) == example.num_events:
                        if example.event[i].parse_event_list(example.events_data[i], self.sparse_mode) is False:
                            return False
                    else:
                        event_lines = len(example.events_data) // example.num_events
                        event_data = ""
                        for idx in range(event_lines):
                            event_data += example.events_data[i * event_lines + idx].strip() + " "
                        if example.event[i].parse_event_list(event_data, self.sparse_mode) is False:
                            return False
                else:
                    if example.event[i].parse_event_list(example_list[j], self.sparse_mode) is False:
                        return False
            # get total time for the example
            example.get_total_time()

        # Scale down the total number of loaded examples to example_set
        self.load_num_examples()

        # set the name of examples for which no name was given
        self.set_example_name()

        # initialize the example_iterator and re order the examples in example list
        self.example_iterator = ExampleIterator.init_example_iterator(self)
        if not self.example_iterator:
            return False
        return True

    def load_num_examples(self):
        """Keep only num_loaded number of examples from all the examples loaded in from the example text file"""
        if self.num_loaded is not None and self.num_loaded < self.num_examples:
            if self.loading_order == "ORDERED":
                while len(self.example) != self.num_loaded:
                    self.example.pop()
            elif self.loading_order == "RANDOM":
                self.first_example = None
                old_example_lst = self.example.copy()
                self.example = []
                index_added = []
                while len(self.example) != self.num_loaded:
                    random_index = random.randint(0, len(old_example_lst) - 1)
                    if random_index not in index_added:
                        self.register_example(old_example_lst[random_index])
                        index_added.append(random_index)

            self.num_examples = self.num_loaded

    def set_example_name(self):
        """Go through the list of examples in this set; if an example doesn't
        have a name, its name will be its index in set.example
        """
        for example in self.example:
            if example.name is None:
                example.name = self.example.index(example)

    def register_example(self, example: Example, new=True):
        """ Add Example E to ExampleSet S and update the attributes of S

        :param example:
        :param new: If the example is new to the list or not
        :type new: bool
        """
        example.next = None
        if not self.first_example:
            self.first_example = example
            self.last_example = example
        else:
            self.last_example.next = example
            self.last_example = example
        if new:
            self.example.append(example)

    def parse_example_set_header_string(self, example_header: str) -> Optional[str]:
        """ Parse through example_header substring and assign the values to S using lookup_list
         :param example_header: substring of the example set file representing an example set header
         :type example_header: str
         :return:
         """
        square_index = float("inf")
        if "[" in example_header:
            square_index = example_header.find("[")
        example_header += "\n"
        lookup_list = ["pre_epoch_proc:", "post_update_proc:", "post_epoch_proc:", "min:", "max:", "grace:", "defI:",
                       "defT:", "actT:", "actI:"]
        for lookup_string in lookup_list:
            if lookup_string in example_header:
                index = example_header.find(lookup_string)
                if index < square_index:
                    find_special_char = re.search(r'\s', example_header[index:]).start() + index
                    value = example_header[index + len(lookup_string): find_special_char].strip()
                    if self.assign_field_values(lookup_string, value) is False:
                        return None
                    example_header = example_header.replace(example_header[index: find_special_char + 1], '')
        return example_header

    def assign_field_values(self, lookup_string: str, value: str) -> bool:
        """ Set the field value of type proc, min, max, grace, defI, defT, actI, actT to their
        respective instance attributes in Event S; if Event is None then assign to ExampleSet S

        :param lookup_string: could be proc, min, max, grace, defI, defT, actI, actT
        :type lookup_string: str
        :param value:
        :type value: str
        :return: Optional error
        """
        reg = re.compile("^[0-9]+([,.][0-9]+)?$")
        if lookup_string == "pre_epoch_proc:":
            self.pre_epoch_proc_name = value
        elif lookup_string == "post_epoch_proc:":
            self.post_epoch_proc_name = value
        elif lookup_string == "post_update_proc:":
            self.post_update_proc_name = value
        elif lookup_string == "min:":
            if reg.match(value):
                self.min_time = float(value)
                # remove the last piece, which is space or "]"
            elif value == "-":
                self.min_time = None
            else:
                return self.parse_error("missing value after \"min:\" in ExampleSet header")
        elif lookup_string == "max:":
            if reg.match(value):
                self.max_time = float(value)
            elif value == "-":
                self.max_time = None
            else:
                return self.parse_error("missing value after \"max:\" in ExampleSet header")
        elif lookup_string == "grace:":
            if reg.match(value):
                self.grace_time = float(value)
            elif value == "-":
                self.grace_time = None
            else:
                return self.parse_error("missing value after \"grace:\" in ExampleSet header")
        elif lookup_string == "defI:":
            if reg.match(value):
                self.default_input = int(value)
            elif value == "-":
                self.default_input = None
            else:
                return self.parse_error("missing value after \"defI:\" in ExampleSet header")
        elif lookup_string == "defT:":
            if reg.match(value):
                self.default_target = int(value)
            elif value == "-":
                self.default_target = None
            else:
                return self.parse_error("missing value after \"defT:\" in ExampleSet header")
        elif lookup_string == "actI:":
            if reg.match(value):
                self.active_input = int(value)
            elif value == "-":
                self.active_input = None
            else:
                return self.parse_error("missing value after \"actI:\" in ExampleSet header")
        elif lookup_string == "actT:":
            if reg.match(value):
                self.active_target = int(value)
            elif value == "-":
                self.active_target = None
            else:
                return self.parse_error("missing value after \"actT:\" in ExampleSet header")
        return True

    def print_out_example_set(self):
        """ Prints out the instance variables of an ExampleSet and of its example_files.
        This function is a work in progress for testing purposes.
        Each new layer of composition is indicated by indent.
        """

        string_output = ""
        string_output += "ExampleSet " + self.name + ": "
        lst = [("fileName", self.file_name), ("pre_epoch_proc", self.pre_epoch_proc_name),
               ("post_update_proc", self.post_update_proc_name),
               ("post_epoch_proc", self.post_epoch_proc_name), ("numEvents", self.num_events),
               ("defI", self.default_input)]
        lst.extend([("actI", self.active_input), ("defT", self.default_target), ("actT", self.active_target)])
        string_output += format_object_line(lst)
        for example_num in range(len(self.example)):
            ex = self.example[example_num]
            string_output += ex.print_out_example(False, 1)
        print(string_output)

    def print_out_examples(self):
        """ Prints out the list of examples in the currently sorted order
        """
        self.example_iterator.print_out_examples()

    def parse_error(self, fmt: str) -> bool:
        """ Prints error message fmt regarding ExampleSet S and return False
        """
        print("loadExample: " + fmt + " of file " + self.file_name)
        return False

    def print_out(self):
        """Print this example set"""
        self.print_out_example_set()

    def write_example_set_to_file(self, file_name):
        """ Writes dense format representation of this example set to file_name
        :param file_name: string of the file to write to.
        :type file_name: str
        """
        output = ";\n"
        file = codecs.open(file_name, "w", 'utf-8')
        output += self.write_example_set_header()
        output += ";\n"
        for ex in self.example:
            output += ex.write_example()
        file.write(output)

    def write_example_set_header(self) -> str:
        """ Returns dense format string representation of this example set header
        """

        s_output = ""
        if self.pre_epoch_proc_name is None:
            s_output += "pre_epoch_proc: " + "-" + "\n"
        else:
            s_output += "pre_epoch_proc: " + self.pre_epoch_proc_name + "\n"
        if self.post_epoch_proc_name is None:
            s_output += "post_epoch_proc: " + "-" + "\n"
        else:
            s_output += "post_epoch_proc: " + self.post_epoch_proc_name + "\n"
        if self.post_update_proc_name is None:
            s_output += "post_update_proc: " + "-" + "\n"
        else:
            s_output += "post_update_proc: " + self.post_update_proc_name + "\n"
        if self.min_time is None:
            s_output += "min: " + "-" + "\n"
        else:
            s_output += "min: " + str(self.min_time) + "\n"
        if self.max_time is None:
            s_output += "max: " + "-" + "\n"
        else:
            s_output += "max: " + str(self.max_time) + "\n"
        if self.grace_time is None:
            s_output += "grace: " + "-" + "\n"
        else:
            s_output += "grace: " + str(self.grace_time) + "\n"
        if self.default_input is None:
            s_output += "defI: " + "-" + "\n"
        else:
            s_output += "defI: " + str(self.default_input) + "\n"
        if self.default_target is None:
            s_output += "defT: " + "-" + "\n"
        else:
            s_output += "defT: " + str(self.default_target) + "\n"
        if self.active_input is None:
            s_output += "actI: " + "-" + "\n"
        else:
            s_output += "actI: " + str(self.active_input) + "\n"
        if self.active_target is None:
            s_output += "actT: " + "-" + "\n"
        else:
            s_output += "actT: " + str(self.active_target) + "\n"
        return s_output


def ignore_commented_lines(example_array: str) -> str:
    """
    Returns new example_array where all text between "#" and the next new line are removed
    :param example_array:
    :return:
    """
    while '#' in example_array:
        index = example_array.find("#")
        find_newline = example_array[index:].find("\n") + index
        example_array = example_array.replace(example_array[index: find_newline + 1], '\n')
    return example_array


# These are helper functions for the self.print_out functions.
def tab(char=1) -> str:
    """Add tab spaces to the print output"""
    return "     " * char


def format_object_line(lst, num_tabs=0) -> str:
    """ This is a helper function for formatting the string in print_out functions.
    """
    string = tab(num_tabs)
    for item in lst:
        string += item[0] + " = " + str(item[1]) + ", "
    return string + "\n"


if __name__ == "__main__":
    pass
