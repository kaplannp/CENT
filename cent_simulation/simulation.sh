# Pipeline Parallel
threads=$1
seqlen_gap=$2

python3 run_sim.py --model Llama2-7B --generate_trace --simulate_trace --process_results --update_csv --num_devices 8 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap
python3 run_sim.py --model Llama2-13B --generate_trace --simulate_trace --process_results --update_csv --num_devices 20 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap
python3 run_sim.py --model Llama2-70B --generate_trace --simulate_trace --process_results --update_csv --num_devices 32 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap

# Model Parallel
python3 run_sim.py --model Llama2-7B --model_parallel --generate_trace --simulate_trace --process_results --update_csv --num_devices 8 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap
python3 run_sim.py --model Llama2-13B --model_parallel --generate_trace --simulate_trace --process_results --update_csv --num_devices 20 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap
python3 run_sim.py --model Llama2-70B --model_parallel --generate_trace --simulate_trace --process_results --update_csv --num_devices 32 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen_gap $seqlen_gap

# Long Context
python3 run_sim.py --model Llama2-70B --generate_trace --simulate_trace --process_results --update_csv --num_devices 32 --run_simulation_max_workers $threads --seqlen 2304 6400 14592 30976

# Scalability Study
python3 scaling_study.py --model Llama2-70B --generate_trace --simulate_trace --process_results --update_csv

# AttAcc/NeuPIM GPT3-175B
python3 run_sim.py --model GPT3-175B --generate_trace --simulate_trace --process_results --update_csv --num_devices 96 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen 128 188 1080 2048

# CXL-PNM OPT-66B
python3 run_sim.py --model OPT-66B --generate_trace --simulate_trace --process_results --update_csv --num_devices 24 --run_simulation_max_workers $threads --generate_trace_max_workers $threads --seqlen 576