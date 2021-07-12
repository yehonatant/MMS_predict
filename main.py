import random
import time
import copy
import pulp
import xpress as xp
from ortools.linear_solver import pywraplp
import json
import os.path
import multiprocessing as mp
import threading as th

def looping_funf(x):
    time.sleep(20)
    return x

def timeout(func, args = (), kwds = {}, timeout = 1, default = None):
    pool = mp.Pool(processes = 1)
    result = pool.apply_async(func, args = args, kwds = kwds)
    try:
        val = result.get(timeout = timeout)
    except mp.TimeoutError:
        pool.terminate()
        return default
    else:
        pool.close()
        pool.join()
        return val


# if __name__ == '__main__':
#     # print(timeout(looping_funf, kwds = {'x': 'Hi'}, timeout = 3, default = 'Bye'))
#     print(timeout(looping_funf, args = (2,), timeout = 2, default = 'Sayonara'))

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# examples_done = 0
# while examples_done < 100:
#     examples_done += 1
#     if examples_done % 10 == 0:
#         print("done with ", examples_done, "examples")


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

def list_missing_data(min_n, max_n, min_m, max_m, min_max_v, max_max_v):
    lst_dirs_files = os.listdir('./Dataset')
    lst_dirs_files = [x for x in lst_dirs_files if x.endswith('.json')]
    lst_dirs_files = [x.strip('.json') for x in lst_dirs_files if x.endswith('.json')]
    lst_dirs_files = [x.split('_') for x in lst_dirs_files]
    lst_dirs_files = [(int(x[0]),int(x[1]),int(x[2])) for x in lst_dirs_files]
    missing_data = []
    for n in range(min_n, max_n+1):
        for m in range(min_m, max_m+1, 10):
            for max_v in range(min_max_v, max_max_v, 50):
                if (n,m,max_v) not in lst_dirs_files:
                    missing_data.append((n,m,max_v))
    return missing_data
missing = list_missing_data(min_n=3, max_n=7, min_m=30, max_m=100, min_max_v=100, max_max_v=500)
for t in missing:
    print(t)