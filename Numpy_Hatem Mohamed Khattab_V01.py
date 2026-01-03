# Mathematics I Lab 04 Numpy tasks
# Hatem Mohamed Khattab

import time
import numpy as np
import scipy as sp
import random
from concurrent.futures import ThreadPoolExecutor
import helpers as h

DATA_TYPE = [('name', 'U16'), ('height', 'f4'), ('class', 'i4')]
ITERATIONS_FAST = 10000
ITERATIONS_SLOW = 100
#####################################################################
#   Write a numpy program to create a structured array from         #
#   given student name, height, class and their data types.         #
#   Then sort the array on height.                                  #
#####################################################################
def student_array(student_data):
    ''' (list) -> NoneType
    Create a numpy array from the input list of tuples, student_data,
    given the data types, DATA_TYPE. Print the original array and
    then sort the array on height and print the sorted array.
    :param student_data: list of tuples containing student data
    '''
    students = np.array(student_data, dtype=DATA_TYPE)

    print("Original Array:")
    print(students)

    
    sorted_students = np.sort(students, order='height')

    print("\nSorted Array (by height):")
    print(sorted_students)

####################################################################
#    Using both Numpy and Scipy                                    #
#    Kindly solve the following equations using Scipy:             #
#                                                                  #
#    ○ x + 2y - 3z + 2w = 30                                       #
#    ○ 2x - 5y + 4z + 9w = 4                                       #
#    ○ -5x + 40y - z - 20w = -6                                    #
#    ○ 5x - 4y - z + 60w = 5                                       #
####################################################################
def solve_equations():
    ''' (None) -> None
        Uses sp.linalg.solve to solve a linear system of equations
        through Gaussian elimination.
        Vectors for the linear system of equations are required to be
        created through numpy arrays.
        Then print the solutions.
    '''
    
    my_switch = input("Would you like to execute the \
        default linear system of equations?(y/n)")
    
    if my_switch == 'y' or my_switch == 'Y':
        coefficients = np.array([
            [1, 2, -3, 2],
            [2, -5, 4, 9],
            [-5, 40, -1, -20],
            [5, -4, -1, 60]
        ])
        
        constants = np.array([30, 4, -6, 5])
    
    if my_switch == 'n' or my_switch == 'N':
        print("Input required for the linear system of equations")
        print("Please enter a comma separated list of numbers\
            representing the coefficients for the variables x,y,z,w,constant")
        raw_1 = input("Please enter the first equation")
        raw_2 = input("Please enter the second equation")
        raw_3 = input("Please enter the third equation")
        raw_4 = input("Please enter the fourth equation")
        
        string_1 = raw_1.replace(" ", "")
        string_2 = raw_2.replace(" ", "")
        string_3 = raw_3.replace(" ", "")
        string_4 = raw_4.replace(" ", "")
        
        list_1 = string_1.split(",")
        list_2 = string_2.split(",")
        list_3 = string_3.split(",")
        list_4 = string_4.split(",")
        
        coefficients_list = [list(map(float, list_1)),
                            list(map(float, list_2)),  
                            list(map(float, list_3)),
                            list(map(float, list_4))]
        
        constants_list = [list_1.pop(), list_2.pop, list_3.pop(), list_4.pop()]
        coefficients = np.array(coefficients_list)
        constants = np.array(constants_list)
        

    
    else:
        print("Invalid input")
        exit(-1)
    
    solutions = sp.linalg.solve(coefficients, constants)
    print("Solutions:")
    print("x ≈ ", solutions.item(0))
    print("y ≈ ", solutions.item(1))
    print("z ≈ ", solutions.item(2))
    print("w ≈ ", solutions.item(3))

####################################################################
#   Implement the numpy vectorized version of the L2 loss function.#
####################################################################
def l2(yhat, y):
    ''' (vector, vector) -> float
    Takes two vectors and returns the l2 loss value.
    :param yhat: vector of size m (predicted labels)
    :param y: vector of size m (true labels)
    :return: value of the l2 loss function defined above
    '''
    
    
    loss = None
    
    return loss

####################################################################
#   Compare the performance of the dot product                     #
#   using parallelism in python                                    #
#   versus numpy_dot_product                                       #
#   versus using a for loop and measure the time needed for each.  #
####################################################################
def compare_dot_product_performance():
    ''' (None) -> None
    Compares the processing time of the dot product 
    using a traditional for loop
    versus a numpy library method 
    versus multithreading in python.
    Takes size of the vectors as input.
    Prints the time measured for each.
    '''
    
    size = int(input("Please enter a natural number for the size of the vectors:"))
    list1 = [random.randint(1, 1000) for _ in range(size)]
    list2 = [random.randint(1, 1000) for _ in range(size)]

    array1 = np.array(list1)
    array2 = np.array(list2)
    
    
    # Measure Parallelized for loop
    start = time.time()
    for _ in range(ITERATIONS_SLOW):
        h.dot_product_parallel(list1, list2)
    avg_parallel = (time.time() - start) / ITERATIONS_SLOW

    # Measure Traditional For Loop
    start = time.time()
    for _ in range(ITERATIONS_SLOW):
        h.dot_product_loop(list1, list2)
    avg_loop = (time.time() - start) / ITERATIONS_SLOW

    # Measure NumPy
    start = time.time()
    for _ in range(ITERATIONS_FAST):
        h.dot_product_numpy(array1, array2)
    avg_numpy = (time.time() - start) / ITERATIONS_FAST


    print(f"\nResults for vector size {size}:")
    print(f"Parallel Dot Time:     {avg_parallel:.8f} seconds")
    print(f"Traditional Loop Time: {avg_loop:.8f} seconds")
    print(f"NumPy Dot Time:        {avg_numpy:.8f} seconds")

if __name__ == '__main__':
    print("Which question would you like to run?\n")
    data_in = input("Enter a number from 1 to 4:")
    
    match(data_in):
        case "1":
            student_data = [
                ('James', 48.5, 1),
                ('Nail', 52.5, 2),
                ('Paul', 42.1, 2),
                ('Pit', 40.1, 1)
            ]
            student_array(student_data)

        case "2":
            solve_equations()
            
        case "3":
            l2()
        
        case "4":
            compare_dot_product_performance()
        
        case _:
            print("Invalid input")
        
    
    print("\nEnd of program")