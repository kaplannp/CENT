import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df_simulation_results = pd.read_csv('cent_simulation/simulation_results.csv')
NeuPIM = pd.read_csv('data/NeuPIM.csv')

seqlen = 188
batch_list = [512, 256, 128, 64]

throughput_list = []
million_tokens_per_dollar_list = []
# TCO: $/hour
# CENT: 96 devices w/ PP=4
# NeuPIM: 8 device w/ PP=4, original paper uses 1 device w/ TP=8 PP=4
CENT_TCO = 8.65
NeuPIM_TCO = 21.97
pp = 4
GPT3_transformer_block_number = 96
transformer_block_number = GPT3_transformer_block_number / pp
for batch in batch_list:
    NeuPIM_throughput = NeuPIM[(NeuPIM['Batch'] == batch)]['Throughput (K Tokens/s)'].iloc[0]
    NeuPIM_CXL_latency = NeuPIM[(NeuPIM['Batch'] == batch)]['CXL Latency (ms)'].iloc[0]
    latency_ms_per_token = batch / NeuPIM_throughput
    latency_ms_per_transformer_block = latency_ms_per_token / transformer_block_number
    new_latency_per_token = (latency_ms_per_transformer_block + NeuPIM_CXL_latency) * transformer_block_number
    new_throughput_K_Tokens_per_second = batch / new_latency_per_token
    throughput_list.append(new_throughput_K_Tokens_per_second)
    million_tokens_per_dollar_list.append(new_throughput_K_Tokens_per_second * 3600 / 1000 / NeuPIM_TCO)
CENT_throughput = df_simulation_results[(df_simulation_results['Model'] == 'GPT3-175B') & (df_simulation_results['Sequence length'] == seqlen)]['Throughput (tokens/s)'].iloc[0] * pp / 1000
throughput_list.append(CENT_throughput)
million_tokens_per_dollar_list.append(CENT_throughput * 3600 / 1000 / CENT_TCO)

import numpy as np
import matplotlib.pyplot as plt

# Sample data
configs = ['Batch\n512', 'Batch\n256', 'Batch\n128', 'Batch\n64', 'Batch\n96']
bar_values = million_tokens_per_dollar_list  # M Tokens/$
scatter_values = throughput_list  # K Tokens/s

# Arrange x positions
x = np.arange(len(configs))
width = 0.4

fig, ax1 = plt.subplots(figsize=(6, 3))

# Create bars for M Tokens/$
bars = ax1.bar(x, bar_values, width, label='M Tokens/$', color='skyblue', edgecolor='black')

# Twin axis for K Tokens/s
ax2 = ax1.twinx()
ax2.scatter(x, scatter_values, color='orange', label='K Tokens/s', zorder=3)
ax1.set_ylim(0, 4.5)
ax2.set_ylim(0, 28)

# Labels and titles
ax1.set_ylabel('M Tokens / $')
ax2.set_ylabel('K Tokens / s')

# Set x-axis labels
ax1.set_xticks(np.arange(0, len(configs), 1) + width / 2)
ax1.set_xlabel("                                NeuPIM                                             CENT")
ax1.set_xticklabels(configs, rotation=45, ha='right')

# Add grid and legend
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Show plot
plt.tight_layout()
if os.path.exists("figures") == False:
    os.mkdir("figures")
plt.savefig('figures/figure_17b.pdf')

df = pd.DataFrame(columns=['System', 'Batch', 'M Tokens/$', 'Throughput (K Tokens/s)'])

for batch in batch_list:
    new_row = {
        'System': 'NeuPIM',
        'Batch': batch,
        'M Tokens/$': million_tokens_per_dollar_list[batch_list.index(batch)],
        'Throughput (K Tokens/s)': throughput_list[batch_list.index(batch)],
    }
    df_new = pd.DataFrame(new_row, index=[0])    
    df = pd.concat([df, df_new], ignore_index=True)
new_row = {
    'System': 'CENT',
    'Batch': batch,
    'M Tokens/$': million_tokens_per_dollar_list[-1],
    'Throughput (K Tokens/s)': throughput_list[-1],
}
df_new = pd.DataFrame(new_row, index=[0])
df = pd.concat([df, df_new], ignore_index=True)

if os.path.exists("figure_source_data") == False:
    os.mkdir("figure_source_data")
df.to_csv('figure_source_data/figure_17b.csv', index=False)