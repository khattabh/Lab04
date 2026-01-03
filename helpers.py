# Mathematics I Lab 04 Numpy tasks
# Hatem Mohamed Khattab

import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Method A: Traditional For Loop
def dot_product_loop(l1, l2):
    result = 0
    for i in range(len(l1)):
        result += l1[i] * l2[i]
    return result

# Method B: NumPy Dot Product
def dot_product_numpy(a1, a2):
    return np.dot(a1, a2)

# Method C: Parallelism
# Helper function to compute dot product for a chunk of the list
def partial_dot(args):
    start, end, l1, l2 = args
    partial_sum = 0
    for i in range(start, end):
        partial_sum += l1[i] * l2[i]
    return partial_sum

def dot_product_parallel(l1, l2, num_workers=2):
    chunk_size = len(l1) // num_workers
    ranges = []
    
    # Divide data into chunks
    for i in range(num_workers):
        start = i * chunk_size
        end = len(l1) if i == num_workers - 1 else (i + 1) * chunk_size
        ranges.append((start, end, l1, l2))
    
    # Execute chunks in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(partial_dot, ranges)
    
    return sum(results)