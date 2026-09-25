from alphagrad.approx.env import _aggregate_samples


# dsnn-dfw.111: the median, the winsorized mean, the top-quartile mean and the mean of these windows all differ.
_EVEN_WINDOWS = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0, 1000.0]
_ODD_WINDOWS = [1.0, 2.0, 3.0, 50.0, 1000.0]


def test_the_latency_of_a_plan_is_the_median_of_its_windows():
    assert float(_aggregate_samples(_EVEN_WINDOWS, want_top_quartile=True)) == 15.0


def test_an_odd_window_count_takes_the_middle_window():
    assert float(_aggregate_samples(_ODD_WINDOWS, want_top_quartile=True)) == 3.0
