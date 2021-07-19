import random
import time
import copy
import pulp
import xpress as xp
from ortools.linear_solver import pywraplp
import json
import os.path
import multiprocessing as mp
import timeout_decorator


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

def timeout(func, args = (), kwds = {}, timeout = 20, default = None):
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

# @timeout_decorator.timeout(30)
def ortools_solver(values, k, timeout=None):
    start = time.time()
    solver = pywraplp.Solver.CreateSolver('SCIP')

    parts = range(k)
    items = range(len(values))
    min_value = solver.IntVar(0.0, sum(values), "min_value")
    vars = [
        [solver.IntVar(0.0, 1, f"x_{item}_{part}")
         for part in parts]
        for item in items
    ]  # vars[i][j] is 1 iff item i is in part j.
    # mms_problem = pulp.LpProblem("MMS_problem", pulp.LpMaximize)
    # mms_problem += min_value  # Objective function: maximize min_value
    for item in items:  # Constraints: each item must be in exactly one part.
        solver.Add(sum([vars[item][part] for part in parts]) == 1)
    for part in parts:  # Constraint: the sum of each part must be at least min_value (by definition of min_value).
        solver.Add(min_value <= sum([vars[item][part] * values[item] for item in items]))

    solver.Maximize(min_value)


    status = solver.Solve()
    if timeout is not None:
        solver.parameters.max_time_in_seconds = timeout
    end = time.time()
    if status == pywraplp.Solver.OPTIMAL:
        return solver.Objective().Value(), end-start
    else:
        print('The problem does not have an optimal solution.')

    print('\nAdvanced usage:')
    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())
    print('Problem solved in %d branch-and-bound nodes' % solver.nodes())

# @timeout_decorator.timeout(30,use_signals=False)
def xpress_solver(values, k, timeout=None):
    start = time.time()
    xp.controls.outputlog = 0
    if timeout is not None:
        xp.controls.maxtime = timeout
    mms_prob = xp.problem(name='mms find', sense=xp.maximize)

    parts = range(k)
    items = range(len(values))
    min_value = xp.var ("min value", vartype=xp.integer, lb=0, ub=sum(values))
    vars = [
        [xp.var(f"x_{item}_{part}", vartype=xp.binary)
         for part in parts]
        for item in items
    ]  # vars[i][j] is 1 iff item i is in part j.
    mms_prob.addVariable(vars, min_value)
    for item in items:  # Constraints: each item must be in exactly one part.
        mms_prob.addConstraint(xp.Sum([vars[item][part] for part in parts]) == 1)
    for part in parts:  # Constraint: the sum of each part must be at least min_value (by definition of min_value).
        mms_prob.addConstraint(min_value <= xp.Sum([vars[item][part] * values[item] for item in items]))
    mms_prob.setObjective(min_value, sense=xp.maximize)
    mms_prob.solve()
    # print("problem status:            ", mms_prob.getSolution ())
    # print("problem status, explained: ", mms_prob.getProbStatusString())
    end = time.time()
    return mms_prob.getSolution(min_value), end-start


def pulp_solver(values: list, k: int)->int:
    start = time.time()
    parts = range(k)
    items = range(len(values))
    min_value = pulp.LpVariable("min_value", cat=pulp.LpContinuous)
    vars = [
        [pulp.LpVariable(f"x_{item}_{part}", lowBound=0, upBound=1, cat=pulp.LpInteger)
        for part in parts]
        for item in items
    ]         # vars[i][j] is 1 iff item i is in part j.
    mms_problem = pulp.LpProblem("MMS_problem", pulp.LpMaximize)
    mms_problem += min_value    # Objective function: maximize min_value
    for item in items:  # Constraints: each item must be in exactly one part.
        mms_problem += (pulp.lpSum([vars[item][part] for part in parts]) == 1)
    for part in parts:  # Constraint: the sum of each part must be at least min_value (by definition of min_value).
        mms_problem += (min_value <= pulp.lpSum([vars[item][part]*values[item] for item in items]))
    pulp.PULP_CBC_CMD(msg=False).solve(mms_problem)
    end = time.time()
    return min_value.value(), end-start


def get_all_k_partitions(my_set, k):
    my_set1 = copy.deepcopy(my_set)
    if len(my_set1) == 0:
        return [[list() for _ in range(k)]]
    else:
        options = []
        item = my_set1.pop()
        options_without_item = get_all_k_partitions(my_set1, k)
        for option_without_item in options_without_item:
            for j in range(k):
                option = copy.deepcopy(option_without_item)
                option[j].append(item)
                options.append(option)
        return options


def maximin_partiotion(M,k):
    partitions = get_all_k_partitions(M, k)
    mms_min = 0
    # min_p = None
    for p in partitions:
        if min([sum(s) for s in p]) >= mms_min:
            mms_min = min([sum(s) for s in p])
            # min_p = p
    min_ps = []
    for p in partitions:
        if min([sum(s) for s in p]) >= mms_min:
            min_ps.append((min([sum(s) for s in p]), p))

    return min_ps


def create_entry(values, mms):
    entry = [
        {'values': values,
         'mms': mms}
    ]
    return entry


def write_examples(n, max_val, m, generation_method, entries):
    path = "./Dataset/" + str(n) + "_" + str(m) + "_" + str(max_val) + "_" + generation_method + ".json"
    all_entries = []
    if os.path.isfile(path):
        with open(path, 'r+') as json_file:
            all_entries += json.load(json_file)
    all_entries += entries
            # json.dump(old_entries, json_file, indent=4)
    with open(path, 'w+') as json_file:
        json.dump(all_entries, json_file, indent=4)


def generate_examples(n, max_val, m, generation_method, examples_count, solver=ortools_solver, sort_reverse=True):
    entries = []
    total_time = 0
    if generation_method == 'uniform':
        examples_done = 0
        while examples_done < examples_count:
            values = [random.randrange(0, max_val + 1) for _ in range(m)]
            values.sort(reverse=sort_reverse)
            time_it_took = 0
            try:
                mms_val, time_it_took = solver(values, n)
            except TimeoutError as e:
                mms_val = None
            if mms_val is not None:
                entries.append(create_entry(values, mms_val))
                total_time += time_it_took
                # time.sleep(0.5)
                examples_done += 1
            else:
                print("timeout", n, m, max_val)
            if examples_done % 10 == 0:
                print("done", examples_done, "examples", n, m, max_val)
        with open("./best solver.txt", "a") as best_solver:
            best_solver.write("\n"+
                              "n="+str(n)+", max_val="+str(max_val)+
                              ", m="+str(m)+", generation_method='uniform':\n"+
                              "\t" + str(solver.__name__)+" = " + str(total_time/examples_count))
        return entries




# k = 8
# m = 80
# mv = 300
# values = [random.randrange(0, mv+1) for _ in range(m)]
#
#
#
# values.sort()
# start = time.time()
# v = ortools_solver(values, k)
# end = time.time()
#
# print("ortools:", v, end-start)
#
# values.sort(reverse=True)
# start = time.time()
# v = ortools_solver(values, k)
# end = time.time()
# print("ortools reverse:", v, end-start)
#
#
# values.sort()
# start = time.time()
# v = xpress_solver(values, k)
# end = time.time()
#
# print("xpress:", v, end-start)
#
# values.sort(reverse=True)
# start = time.time()
# v = xpress_solver(values, k)
# end = time.time()
# print("xpress reverse:", v, end-start)





ns = [8]

ms = [80]
max_vals = [300]

generation_methods=['uniform']
for n in ns:
    for max_val in max_vals:
        for m in ms:
            for generation_method in generation_methods:
                b=random.randint(0,1)
                entries = generate_examples(n=n, max_val=max_val, m=m, generation_method=generation_method,
                                            examples_count=100,
                                            solver=xpress_solver, sort_reverse=bool(b))
                write_examples(n=n, max_val=max_val, m=m, generation_method=generation_method, entries=entries)
                print("done "+ str(n) + ", "+str(m)+", "+str(max_val))
