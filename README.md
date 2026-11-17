# ProcessBench

📄 [**[paper]**](https://huggingface.co/papers/2412.06559) 🤗 [**[data]**](https://huggingface.co/datasets/Qwen/ProcessBench)

This is the official repository for **ACL 2025** paper "[ProcessBench: Identifying Process Errors in Mathematical Reasoning](https://huggingface.co/papers/2412.06559)"

If you find this work relevant or helpful to your work, please kindly cite us:

```latex
@inproceedings{
  processbench,
  title={ProcessBench: Identifying Process Errors in Mathematical Reasoning}, 
  author={
    Chujie Zheng and Zhenru Zhang and Beichen Zhang and Runji Lin and Keming Lu and
    Bowen Yu and Dayiheng Liu and Jingren Zhou and Junyang Lin
  },
  booktitle={The 63rd Annual Meeting of the Association for Computational Linguistics
},
  year={2025}
}
```

## Data Usage

You can use the following code to preview the ProcessBench data:

```python
import json
from datasets import load_dataset

dataset = load_dataset('Qwen/ProcessBench', split='gsm8k')
print(json.dumps(dataset[0], indent=2))

# Expected output:
"""
{
  "id": "gsm8k-0",
  "generator": "Qwen2-7B-Instruct",
  "problem": "Sue lives in a fun neighborhood...",
  "steps": [
    "To find out how many more pink plastic flamingos were out than...",
    ...
  ],
  "final_answer_correct": false,
  "label": 1
}
"""
```

## Evaluation

You can refer to the [code](./code) folder for the evaluation code and the prompt templates we use in this work

### Codex backend (Harbor parity)

This fork adds `--backend codex` to `code/run_eval.py` for parity experiments with the [Harbor](https://github.com/harbor-framework/harbor) ProcessBench adapter ([harbor-framework/adapters#70](https://github.com/harbor-framework/adapters/pull/70)). The default `--backend vllm` path behaves as before. The Harbor-side reproduction steps are in the adapter README.

For each item, the codex backend appends one sentence to the critique prompt (``Write your final answer to the file `answer.txt` in the current working directory, in the form \boxed{index}.``), runs `codex exec` with the same flags as Harbor's codex agent in a fresh working directory and `CODEX_HOME`, and reads `answer.txt`. Answer extraction and F1 are the unchanged upstream code. The prompt is byte-identical to the Harbor task's `instruction.md`.

```bash
cd code
export OPENAI_API_KEY=...
python run_eval.py --backend codex --model <model> \
    --codex_config /path/to/config.toml \
    --data_dir ../data --ids_file parity_ids.txt \
    --n_concurrent 10
```

`--codex_config` (optional) is copied to `CODEX_HOME/config.toml` for every item, e.g. to set the provider, base URL, or reasoning effort. `--data_dir` loads local `<subset>.json` files instead of the HF hub, and `--ids_file parity_ids.txt` restricts the run to the 200-item Harbor parity subset.

Requires `codex` on `PATH` and `numpy`. Outputs go to `outputs/codex_<model>/`: the usual `<subset>_{error,correct}.jsonl`, per-item results in `codex_items/` (reused when a run is resumed), and codex logs and session rollouts in `codex_logs/`.
