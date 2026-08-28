

from PyLens.examples.example_set import *

test_example_set = ExampleSet.initialize_example_set(None, False, "test", "rand10x40.ex", [], [], 1, 1, 1, 1)
assert len(test_example_set.example) == 40

test_example = test_example_set.example[0]
assert len(test_example.event) == 1

test_encoder_dense = ExampleSet.initialize_example_set(None, False, "test", "encoder.dense.ex", [], [], 1, 1, 1, 1)
test_encoder_sparse = ExampleSet.initialize_example_set(None, False, "test", "encoder.sparse.ex", [], [], 1, 1, 1, 1)
assert test_encoder_dense == test_encoder_sparse
print("passed")
