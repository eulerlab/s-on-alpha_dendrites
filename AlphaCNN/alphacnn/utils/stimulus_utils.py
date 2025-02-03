def get_stimulus_shape(pixel_size: int, stimulus_size_x: int, stimulus_size_y: int, stimulus_size_t: int = 1):
    assert stimulus_size_x % pixel_size == 0
    assert stimulus_size_y % pixel_size == 0

    if stimulus_size_t <= 1:
        stimulus_shape = (stimulus_size_x // pixel_size, stimulus_size_y // pixel_size, 1)
    else:
        stimulus_shape = (stimulus_size_x // pixel_size, stimulus_size_y // pixel_size, int(stimulus_size_t))
    return stimulus_shape
