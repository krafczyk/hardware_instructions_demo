import subprocess
import re
import pandas as pd
import numpy as np

res_matcher = re.compile("Took ([0-9]*\.*[0-9]*) milliseconds")

def run_test_1(test_name, test_size):
    result = subprocess.check_output(["./build/bin/"+test_name, '-size', str(test_size)])
    return result

def run_tests_1(test_name, test_size, num, df):
    for i in range(num):
        res = run_test_1(test_name, test_size).decode('utf-8')
        match_res = res_matcher.search(res)
        if match_res is None:
            print(f"Warning test: {test_name},{test_size} had no suitable output.")
            df = df.append({'name': test_name, 'size': int(test_size), 'time': np.nan}, ignore_index=True);
        else:
            df = df.append({'name': test_name, 'size': int(test_size), 'time': float(match_res.group(1))}, ignore_index=True)
    return df

if __name__ == "__main__":
    result_df = pd.DataFrame(columns=['name', 'size', 'time'])
    result_df['name'].astype(str)
    result_df['size'].astype(np.int)
    result_df['time'].astype(np.float)

    Num = 20

    test_size = 512

    test_names = [
        "loop1_O3_nehalem_float",
        "loop1_O3_broadwell_float",
        "loop1_mkl_O3_nehalem_float",
        "loop1_mkl_O3_broadwell_float",
        "loop1_openblas_O3_nehalem_float",
        "loop1_openblas_O3_broadwell_float",
    ]

    for test_name in test_names:
        result_df = run_tests_1(test_name, test_size, Num, result_df)
    result_df.to_csv("test_results.csv")

    result_df = result_df.groupby(['name', 'size']).agg({'time': [np.mean, np.std]})
    print(result_df)
