def normalize(arr):
    min_val = arr.min()
    max_val = arr.max()
    return (arr - min_val)/(max_val - min_val), min_val, max_val

def denormalize(arr, min_val, max_val):
    return (arr * (max_val - min_val)) + min_val

def normalize_val(val, min_val, max_val):
    return (val - min_val)/(max_val - min_val)