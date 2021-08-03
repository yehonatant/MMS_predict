import os.path

def list_missing_data(min_n, max_n, min_m, max_m, min_max_v, max_max_v):
    lst_dirs_files = os.listdir('./Dataset')
    lst_dirs_files = [x for x in lst_dirs_files if x.endswith('.json')]
    lst_dirs_files = [x.strip('.json') for x in lst_dirs_files if x.endswith('.json')]
    lst_dirs_files = [x.split('_') for x in lst_dirs_files]
    lst_dirs_files = [(int(x[0]),int(x[1]),int(x[2])) for x in lst_dirs_files]
    missing_data = []
    for n in range(min_n, max_n+1):
        for m in range(min_m, max_m+1, 10):
            for max_v in range(min_max_v, max_max_v+1, 50):
                if (n,m,max_v) not in lst_dirs_files:
                    missing_data.append((n,m,max_v))
    return missing_data
missing = list_missing_data(min_n=3, max_n=8, min_m=30, max_m=100, min_max_v=100, max_max_v=350)
for t in missing:
    print(t)