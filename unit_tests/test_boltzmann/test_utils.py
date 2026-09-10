import math

def check_reports(net):
    progress_stats = net.stats_plotter.progress_stats
    for x in progress_stats['error']:
        assert not math.isnan(x)
        assert x >= 0

def check_difference(x, y, buffer=0.001):
    assert abs(x-y) < buffer
