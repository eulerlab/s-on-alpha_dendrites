import numpy as np


def get_offsets(stack_nx, stack_ny, stack_pixel_size_um, field_cpos_stack_xy, roi_cpos_stack_xy=None):
    stack_offset = np.array([stack_nx / 2, stack_ny / 2])
    stack_offset_um = stack_offset * stack_pixel_size_um

    morph_field_offset_um = (stack_offset - field_cpos_stack_xy) * stack_pixel_size_um

    if roi_cpos_stack_xy is not None:
        roi_field_offset_um = (roi_cpos_stack_xy - field_cpos_stack_xy) * stack_pixel_size_um
    else:
        roi_field_offset_um = None

    return stack_offset_um, morph_field_offset_um, roi_field_offset_um
