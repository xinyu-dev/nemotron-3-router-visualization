# nemotron-3-router-visualization


## Usage

1. Clone repo
```bash
git clone https://github.com/xinyu-dev/nemotron-3-router-visualization.git
```

2. You can just use the Nemo-RL enviornment to run the script. For example: 

```bash
# clone repo
git clone https://github.com/NVIDIA-NeMo/RL.git && cd RL

# fetch submodules
git submodule update --init --recursive

# create uv && install
uv venv && uv sync 

# activate uv env
source .venv/bin/activate
```

2. Extract the router scores 

```bash
python extract_router_data.py \
--model /workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
--prompt "Observations of structures located at a distance of about 2.1 gigaparsecs (2.1 Gpc) are being carried out. The detected absorption line energy equivalent is about 3.9 micro electron volts (3.9 * 10^-6 eV). What is most likely to be observed with this absorption line in the Milky Way?A. Warm atomic interstellar medium. B. Cold molecular interstellar medium. C. Cold atomic interstellar medium. D. Warm molecular interstellar medium." \
--output output/router_data.npz
```

- `model`: change it to the location of the HF model weights
- `prompt`: your prompt
- `output`: output file location. Need to save as a `npz` file


The `npz` file contains the following: 

```bash
Output shape
------------
RouterData.scores           : np.ndarray  [seq_len, num_moe_layers, n_routed_experts]
RouterData.topk_indices     : np.ndarray  [seq_len, num_moe_layers, top_k]
RouterData.topk_weights     : np.ndarray  [seq_len, num_moe_layers, top_k]
RouterData.tokens           : list[str]   length seq_len
RouterData.moe_layer_indices: list[int]   length num_moe_layers

Where
  seq_len          = number of input tokens
  num_moe_layers   = number of MoE blocks in the network (23 for this config for Nemotron-3 Nano)
  n_routed_experts = 128 (number of experts, 128 for this config for Nemotron-3 Nano)
  top_k            = 6 (number of selected experts per token, 6 for this config for Nemotron-3 Nano)
```


3. Visaulize

To generate visualizaiton

```bash
python visualize_router.py --input output/router_data.npz
```

