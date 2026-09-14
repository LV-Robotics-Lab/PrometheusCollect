import numpy as np

from prometheus.data.episode_plots import (
    insert_nan_at_time_gaps,
    time_gaps,
)


def test_time_gaps_detects_recording_pause():
    t_s = np.array([0.0, 0.033, 0.066, 6.5, 6.533], dtype=np.float64)
    gaps = time_gaps(t_s)
    assert gaps == [(0.066, 6.5)]


def test_insert_nan_at_time_gaps_breaks_line():
    t_s = np.array([0.0, 0.033, 0.066, 6.5, 6.533], dtype=np.float64)
    values = np.array([1.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float64)

    plot_t, plot_v = insert_nan_at_time_gaps(t_s, values, gap_threshold_s=0.2)

    assert plot_t.shape[0] == plot_v.shape[0] == 6
    assert np.isnan(plot_t[3])
    assert np.isnan(plot_v[3])
    assert plot_v[2] == 1.0
    assert plot_v[4] == 2.0
