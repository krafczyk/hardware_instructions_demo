import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#plt.set_cmap('Pastel1')
mpl.style.use('seaborn')

# Get color cycle
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Load data
results_df = pd.read_csv('test_results.csv', dtype={'name':str, 'size': int, 'time': np.float})
grp_df = results_df.groupby(['name', 'size']).agg({'time': [np.mean, np.std]})

type_str = ['_', '_mkl_', '_openblas_']
legend_str = ['Plain Loop', 'MKL', 'OpenBLAS']
cpu_str = ['nehalem', 'ivybridge', 'skylake', 'knl']

def get_full_name(type_, cpu):
    return f"loop1{type_}O3_{cpu}_float"

barWidth = 1/(len(cpu_str)+1)

bar_height = {}
bar_var = {}
bar_xs = {}
for i in range(len(cpu_str)):
    cpu = cpu_str[i]
    bar_height[cpu] = []
    bar_var[cpu] = []
    if i == 0:
        bar_xs[cpu] = np.arange(len(type_str))
    else:
        bar_xs[cpu] = [x+barWidth for x in bar_xs[cpu_str[i-1]]]

    for type_ in type_str:
        full_name = get_full_name(type_, cpu)
        bar_height[cpu].append(grp_df.loc[(full_name,1024)].loc[('time', 'mean')])
        bar_var[cpu].append(grp_df.loc[(full_name,1024)].loc[('time', 'std')])

for i in range(len(cpu_str)):
    cpu = cpu_str[i]
    plt.bar(bar_xs[cpu], bar_height[cpu], yerr=bar_var[cpu], width=barWidth, color=color_cycle[i], label=cpu_str[i], edgecolor='white')

plt.xlabel('Addition Method', fontweight='bold')
plt.ylabel('Addition Time [ms]', fontweight='bold')
plt.xticks([r+barWidth for r in range(len(type_str))], legend_str, fontweight='bold')
plt.yticks(fontweight='bold')
plt.title('Time to add two 1GB arrays', fontweight='bold')

plt.legend(prop={'weight': 'bold'})

plt.savefig('test_results_plot.png')
