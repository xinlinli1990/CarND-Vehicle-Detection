import sys
import os

from parameter_tuning import run

def read_nomad_input_file(argv):
    with open(argv[1], 'r') as nomad_input:
        nomad_input_data = nomad_input.read().splitlines()[0].split(" ")
    return nomad_input_data

input_parameters = read_nomad_input_file(sys.argv)
i_cspace, spatial, histbin, orient, pix_per_cell, cell_per_block, hog_channel = input_parameters

# print(learning_rate)
# print(data_augmentation_target)
# print(dropout_keep_prob)
# print(l2_reg_const_param)

predict_accu = run(int(i_cspace),
                   int(spatial),
                   int(histbin),
                   int(orient),
                   int(pix_per_cell),
                   int(cell_per_block),
                   int(hog_channel))

print("{0:.4f}".format(-1 * predict_accu))