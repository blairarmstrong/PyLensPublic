import test_boltz_big, test_boltz_xor, test_boltz_complete2, test_boltz_on_off

def main():
    test_boltz_xor.boltz_xor_unit_test()
    test_boltz_big.boltz_big_unit_test()
    test_boltz_complete2.boltz_complete2_unit_test()
    test_boltz_on_off.boltz_multi_event_unit_test()

if __name__ == '__main__':
    main()