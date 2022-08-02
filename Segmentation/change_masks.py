def change_masks(input_data):    
    input_data[input_data == 3] = 0
    input_data[input_data == 4] = 0
    input_data[input_data == 5] = 0
    input_data[input_data == 6] = 2
    input_data[input_data == 7] = 0
    return input_data