import math


def assert_increasing(arr):
    assert arr == sorted(arr)

def assert_decreasing(arr, buffer=0.1):
    for i in range(len(arr)-1):
        assert arr[i] - arr[i-1] <= buffer

def check_reports(net):
    progress_stats = net.stats_plotter.progress_stats
    for x in progress_stats['error']:
        assert not math.isnan(x)
        assert x >= 0
    # assert_decreasing(progress_stats['remaining_time'])
    assert_decreasing(progress_stats['error'], 0.1)

def check_difference(x, y, buffer=0.1):
    assert abs(x-y) < buffer
