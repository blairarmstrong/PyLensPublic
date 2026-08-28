import test_xor as xor
import test_encoder as encoder
import test_negation as negation
import sys
sys.path.insert(0, ".")
from PyLens.simulator import Simulator

# sim = Simulator(name="sim")

# result = True
# result1 = xor.xor_test(sim)
# sim.delete_all_nets()
result3 = negation.negation_test()
# sim.delete_all_nets()
# result2 = encoder.encoder_test()

result = result3

if result:
    print("Final Passed")
    exit(0)
else:
    print("Final Failed")
    exit(1)
