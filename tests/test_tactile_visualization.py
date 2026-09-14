from __future__ import annotations

import numpy as np

from prometheus.visualization.tactile import (
    DZ_CHANNEL,
    TACTILE_CHANNELS,
    TACTILE_GRID_HEIGHT,
    TACTILE_GRID_WIDTH,
    dz_grid_to_bgr,
    is_tactile_flow_stream,
    normalize_grid_to_u8,
    pointcloud_msg_to_dz_bgr,
    pointcloud_msg_to_dz_grid,
)


def _synthetic_pointcloud(*, dz_value: float = 0.0) -> np.ndarray:
    flat = np.zeros((TACTILE_GRID_HEIGHT * TACTILE_GRID_WIDTH, TACTILE_CHANNELS), dtype=np.float32)
    flat[:, DZ_CHANNEL] = dz_value
    return flat


def test_is_tactile_flow_stream():
    assert is_tactile_flow_stream("left_wrist_0_tactile_flow")
    assert not is_tactile_flow_stream("left_wrist_0_tactile_difference")


def test_pointcloud_msg_to_dz_grid_shape_and_channel():
    flat = _synthetic_pointcloud()
    flat[0, DZ_CHANNEL] = 1.5
    flat[-1, DZ_CHANNEL] = -2.0

    grid = pointcloud_msg_to_dz_grid(flat)

    assert grid.shape == (TACTILE_GRID_HEIGHT, TACTILE_GRID_WIDTH)
    assert grid[0, 0] == 1.5
    assert grid[-1, -1] == -2.0


def test_normalize_grid_to_u8_percentile_scaling():
    grid = np.array(
        [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
        ],
        dtype=np.float32,
    )

    normalized = normalize_grid_to_u8(grid, percentile_low=0.0, percentile_high=100.0)

    assert normalized.dtype == np.uint8
    assert int(normalized.min()) == 0
    assert int(normalized.max()) == 255
    assert int(normalized[0, 0]) == 0
    assert int(normalized[-1, -1]) == 255


def test_dz_grid_to_bgr_outputs_jet_colormap_image():
    grid = np.linspace(0.0, 1.0, TACTILE_GRID_HEIGHT * TACTILE_GRID_WIDTH, dtype=np.float32)
    grid = grid.reshape(TACTILE_GRID_HEIGHT, TACTILE_GRID_WIDTH)

    image = dz_grid_to_bgr(grid)

    assert image.shape == (TACTILE_GRID_HEIGHT, TACTILE_GRID_WIDTH, 3)
    assert image.dtype == np.uint8
    assert image.max() > 0


def test_pointcloud_msg_to_dz_bgr_end_to_end():
    flat = _synthetic_pointcloud()
    flat[:, DZ_CHANNEL] = np.linspace(-1.0, 1.0, flat.shape[0], dtype=np.float32)

    image = pointcloud_msg_to_dz_bgr(flat)

    assert image.shape == (TACTILE_GRID_HEIGHT, TACTILE_GRID_WIDTH, 3)
    assert image.dtype == np.uint8
