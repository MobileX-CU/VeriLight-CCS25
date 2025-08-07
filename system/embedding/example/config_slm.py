"""This file is a modified copy of config.py generated for the SLM and copied to the Pi at the start of the session."""
fade = False
scale_factor_disp = 1.6
N = 29
buffer_space = 12
slm_W = 640
slm_H = 360
slm_W = 640
localization_N = 2
embedding_window_duration  = 4
localization_frequency  = 6
frequency = 3
target_channel = sum
"""
Do not touch! The driver automatically copies it to the Raspberry Pi at startup and uses it to configure the SLM. You don't need to worry about this though
"""


max_cells_W = int((slm_W ) / (N + buffer_space))
max_cells_H = int((slm_H + buffer_space) / (N + buffer_space))

if localization_N is not None:
    max_cells = max_cells_H * max_cells_W - 4*localization_N**2
    num_pilot_cells = max_cells_W * 2 + max_cells_H * 2 - 4 - ((2*localization_N - 1)*4)
    num_info_cells = (max_cells_H * max_cells_W) - num_pilot_cells - 4*(localization_N**2)
    reserved_localization_cells = []
else:
    max_cells = max_cells_H * max_cells_W 
    num_pilot_cells = max_cells_W * 2 + max_cells_H * 2 - 4
    num_info_cells = (max_cells_H * max_cells_W) - num_pilot_cells
    reserved_localization_cells = [] #"r-c" index format. leave empty if don't want any reserved.
    count = 0
    for c in range(max_cells_W):
        if count % 2 == 0:
            reserved_localization_cells.append(f"{0}-{c}")
        count += 1
    for r in range(1, max_cells_H):
        if count % 2 == 0:
            reserved_localization_cells.append(f"{r}-{max_cells_W-1}")
        count += 1
    for c in range(max_cells_W-2, -1, -1):
        if count % 2 == 0:
            reserved_localization_cells.append(f"{max_cells_H-1}-{c}")
        count += 1
    for r in range(max_cells_H-2, 0, -1):
        if count % 2 == 0:
            reserved_localization_cells.append(f"{r}-0")
        count += 1

