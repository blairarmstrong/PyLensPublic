import random
from typing import Optional, List
from .example import Example


class ExampleIndex:
    """ExampleIndex Object. Stores the next and prev ExampleIndex and the value of the current ExampleIndex object
    tracked the iterator class which is basically the index of the current example in the example array
    """
    next = None  # ExampleIndex
    prev = None  # ExampleIndex
    value: int

    def __init__(self, value: int):
        self.next = None
        self.prev = None
        self.value = value


class ExampleIterator:
    """This class iterates over the examples in an ExampleSet object and keeps track of the current, previous and next example at each iteration. It does not changes original order of the examples in the example array, instead keeps the track of their indices in the array and is also responsible for ordering the example indices based on the value of mode parameter passed in the ExampleSet constructor. Also resets itself once it reaches the end of the array.
    """
    curr: ExampleIndex
    iter_list: List[ExampleIndex]
    num_examples: int
    index_list: List[int]

    def __init__(self, example_set):
        self.example_set = example_set
        self.example = example_set.example
        self.num_examples = example_set.num_examples
        self.index_list = []
        self.iter_list = []

    @classmethod
    def init_example_iterator(cls, example_set) -> Optional['ExampleIterator']:
        """initialize this Iterator class and re sort the example list from the example_set parameter"""
        iterator = cls(example_set)
        if iterator.reset_example_list() is False:
            return None
        return iterator

    def link_up_index_list(self):
        """
        Load in and link up ExampleIndex object in the iter_list instance variable which will be used to keep track of
        the curr, prev and next examples in the ordered examples array
        :return: None
        """
        self.iter_list = []
        for i in range(self.num_examples):
            self.iter_list.append(ExampleIndex(self.index_list[i]))

        for i in range(1, self.num_examples):
            self.iter_list[i - 1].next = self.iter_list[i]
        for i in range(self.num_examples - 2, -1, -1):
            self.iter_list[i + 1].prev = self.iter_list[i]
        self.curr = self.iter_list[0]

    def iterate_example(self) -> Optional[Example]:
        """ Returns the example at index self.curr_ex_index and increments self.curr_ex_index
        of self.example_sorted. If the index is the last index of the list, re-sort the list.
        :return: next example
        :rtype: Optional[Example]
        """
        example = self.example[self.curr.value]
        if self.curr.next is None:
            self.reset_example_list()
        else:
            self.curr = self.curr.next
        return example

    def current_example(self) -> Example:
        """ Returns current example
        """
        return self.example[self.curr.value]

    def first_example(self) -> Example:
        """
        Return the first example in the ordered example array and return None otherwise.
        :return: Example
        """
        return self.example[self.iter_list[0].value]

    def last_example(self) -> Example:
        """
        Return the last example in the example array and return None otherwise.
        :return: Example
        """
        return self.example[self.iter_list[-1].value]

    def next_example(self) -> Optional[Example]:
        """
        Return the next example in the example array and return None otherwise.
        :return: Example
        """
        if self.curr.next is None:
            return None
        return self.example[self.curr.next.value]

    def prev_example(self) -> Optional[Example]:
        """
        Return the previous example in the example array and return None otherwise.
        :return: Example
        """
        if self.curr.prev is None:
            return None
        return self.example[self.curr.prev.value]

    def reset_example_list(self):
        """ Re-sort the example list according to mode and updates first_example,
        last_example and each example.next accordingly.
        """
        if not self.example:
            return False
        if self.sort_examples() is False:
            return False
        self.link_up_index_list()
        return True

    def print_out_examples(self):
        """
        Print the examples in the mode they are ordered in
        :return: None
        """
        example_string = ""
        for example in self.iter_list:
            example_string += " -> "
            example_string += self.example[example.value].name + " i=" + str(example)
        print(example_string)

    def sort_examples(self) -> Optional[bool]:
        """
        Populate ``self.index_list`` with example indices sorted according to
        the current ``sort_mode`` of the example set.

        The following modes determine how examples are selected:

        ORDERED
            Examples are returned in the order they appear in the file.

        RANDOMIZED
            Examples are selected at random *with* replacement. Each example has
            equal probability. This differs from PERMUTED (no replacement) and
            PROBABILISTIC (uses frequency).

        PERMUTED
            Examples are selected at random *without* replacement. A new order is
            computed for each pass through the set.

        PROBABILISTIC
            Examples are selected based on their specified frequency. Frequencies
            are normalized to form a probability distribution. Examples without an
            explicit frequency are treated as having frequency 1.0.

        PIPE
            Examples are read sequentially from a pipe. If the pipe is exhausted
            and ``pipeLoop`` is True, the pipe is reopened automatically. If the
            example set contains both stored examples and an open pipe, you can
            switch between them by changing from PIPE mode to another mode.

        CUSTOM
            A custom procedure ``chooseExample`` is called to generate the index
            of the next example.

        Returns:
            Optional[bool]: ``None`` for normal modes, or the result of
            ``parse_error`` if an invalid sort mode is encountered.
        """
        mode = self.example_set.sort_mode
        self.index_list = []
        if mode == "ORDERED":
            for i in range(self.num_examples):
                self.index_list.append(i)
            return
        elif mode == "RANDOMIZED":
            for _ in range(self.num_examples):
                random_index = random.randint(0, self.num_examples - 1)
                self.index_list.append(random_index)
            return
        elif mode == "PERMUTED":
            for i in range(self.num_examples):
                self.index_list.append(i)
            random.shuffle(self.index_list)
            return
        elif mode == "PROBABILISTIC":
            total_freq = 0.0
            freq_cum = [0.0]
            # cumulative frequency of all previous examples parsed. the greater the frequency
            # of an example, the greater the increment over the previous value.
            for example in self.example:
                if isinstance(example.frequency, float):
                    total_freq += example.frequency
                else:
                    total_freq += 0.0
                    # return self.parse_error("error reading frequency")
                freq_cum.append(total_freq)
            for _ in range(self.num_examples):
                random_choice = random.random() * total_freq
                index = 0
                while freq_cum[index + 1] < random_choice:
                    index += 1
                self.index_list.append(index)
            return
        elif mode == "CUSTOM":
            return
        else:
            return self.example_set.parse_error("invalid sort mode")
