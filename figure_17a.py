import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df_simulation_results = pd.read_csv('cent_simulation/simulation_results.csv')
AttAcc = pd.read_csv('data/AttAcc.csv')

seqlen_list = ["In=128 Out=128", "In=128 Out=2048", "In=2048 Out=128", "In=2048 Out=2048"]
seqlen_l = [128, 1080, 1080, 2048]

throughput_list = []
million_tokens_per_dollar_list = []
# TCO: $/hour
# CENT: 96 devices
# AttAcc: 8 A100 with HBM3-PIM
CENT_TCO = 2.16
AttAcc_TCO = 7.36
for seqlen in seqlen_list:
    AttAcc_throughput = AttAcc[(AttAcc['Seqlen'] == seqlen)]['Throughput (tokens/s)'].iloc[0]
    throughput_list.append(AttAcc_throughput)
    million_tokens_per_dollar_list.append(AttAcc_throughput * 3600 / 1000 / AttAcc_TCO)
    CENT_throughput = df_simulation_results[(df_simulation_results['Model'] == 'GPT3-175B') & (df_simulation_results['Sequence length'] == seqlen_l[seqlen_list.index(seqlen)])]['Throughput (tokens/s)'].iloc[0]
    throughput_list.append(CENT_throughput)
    million_tokens_per_dollar_list.append(CENT_throughput * 3600 / 1000 / CENT_TCO)

throughput_list = [i/1000 for i in throughput_list]
million_tokens_per_dollar_list = [i/1000 for i in million_tokens_per_dollar_list]

import numpy as np
import matplotlib.pyplot as plt

# Sample data
categories = ['AttAcc', 'CENT'] * 4
configs = ['In 128\nOut 128', 'In 128\nOut 2K', 'In 2K\nOut 128', 'In 2K\nOut 2K']
num_groups = len(configs)
bar_values = million_tokens_per_dollar_list  # M Tokens/$
scatter_values = throughput_list  # K Tokens/s

# Arrange x positions
x = np.arange(len(categories))
width = 0.4

fig, ax1 = plt.subplots(figsize=(6, 3))

# Create bars for M Tokens/$
bars = ax1.bar(x, bar_values, width, label='M Tokens/$', color='skyblue', edgecolor='black')

# Twin axis for K Tokens/s
ax2 = ax1.twinx()
ax2.scatter(x, scatter_values, color='orange', label='K Tokens/s', zorder=3)
ax1.set_ylim(0, 4.5)
ax2.set_ylim(0, 10)

# Labels and titles
ax1.set_ylabel('M Tokens / $')
ax2.set_ylabel('K Tokens / s')

# Set x-axis labels
ax1.set_xticks(np.arange(0, len(categories), 2) + width / 2)
ax1.set_xlabel("    ".join(categories))
ax1.set_xticklabels(configs, rotation=45, ha='right')

# Add grid and legend
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Show plot
plt.tight_layout()
if os.path.exists("figures") == False:
    os.mkdir("figures")
plt.savefig('figures/figure_17a.pdf')

df = pd.DataFrame(columns=['System', 'Seqlen', 'M Tokens/$', 'Throughput (K Tokens/s)'])

for seqlen in seqlen_list:
    new_row = {
        'System': 'AttAcc',
        'Seqlen': seqlen,
        'M Tokens/$': million_tokens_per_dollar_list[seqlen_list.index(seqlen)*2],
        'Throughput (K Tokens/s)': throughput_list[seqlen_list.index(seqlen)*2],
    }
    df_new = pd.DataFrame(new_row, index=[0])    
    df = pd.concat([df, df_new], ignore_index=True)
    new_row = {
        'System': 'CENT',
        'Seqlen': seqlen,
        'M Tokens/$': million_tokens_per_dollar_list[seqlen_list.index(seqlen)*2+1],
        'Throughput (K Tokens/s)': throughput_list[seqlen_list.index(seqlen)*2+1],
    }
    df_new = pd.DataFrame(new_row, index=[0])
    df = pd.concat([df, df_new], ignore_index=True)

if os.path.exists("figure_source_data") == False:
    os.mkdir("figure_source_data")
df.to_csv('figure_source_data/figure_17a.csv', index=False)