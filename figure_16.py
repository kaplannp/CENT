import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df_simulation_results = pd.read_csv('cent_simulation/simulation_results.csv')
CXL_PNM = pd.read_csv('data/CXL_PNM.csv')

throughput_list = []
device_list = [1, 8, 32]
for device in device_list:
    CXL_PNM_throughput = CXL_PNM[(CXL_PNM['Devices'] == device)]['Throughput (tokens/s)'].iloc[0]
    throughput_list.append(CXL_PNM_throughput)

df = df_simulation_results[(df_simulation_results['Model'] == 'OPT-66B') & (df_simulation_results['Device number'] == 24)]
throughput_list.append(df['Throughput (tokens/s)'].mean().item())

# print(throughput_list)

throughput_list = [i/1000 for i in throughput_list]

x_lable = ["1", "8", "32", "24"]
# Plot
plt.figure(figsize=(5, 5))
plt.bar(x_lable, throughput_list, color='skyblue', edgecolor='black')

# Labels
plt.xlabel("Device Number", fontsize=12)
plt.ylabel("K Tokens/s", fontsize=12)

# Formatting
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

if os.path.exists("figures") == False:
    os.mkdir("figures")
plt.savefig('figures/figure_16.pdf')

df = pd.DataFrame(columns=['Throughput (K Tokens/s)'])

for i in range(len(throughput_list)):
    new_row = {
        'Throughput (K Tokens/s)': throughput_list[i],
    }
    df_new = pd.DataFrame(new_row, index=[0])    
    df = pd.concat([df, df_new], ignore_index=True)

if os.path.exists("figure_source_data") == False:
    os.mkdir("figure_source_data")
df.to_csv('figure_source_data/figure_16.csv', index=False)