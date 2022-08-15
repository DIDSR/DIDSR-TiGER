def change_masks(input_data):    
    # This function relables the input_mask to segment the tissue into "rest", "tumor", and "stroma" classes.
    # input_data: numpy array of size (N,M)
    input_data[input_data == 3] = 0
    input_data[input_data == 4] = 0
    input_data[input_data == 5] = 0
    input_data[input_data == 6] = 2
    input_data[input_data == 7] = 0
    return input_data
