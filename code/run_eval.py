import argparse
import numpy as np
import os
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import re

# Appended to the critique prompt for the codex backend (same sentence as the Harbor adapter).
CODEX_OUTPUT_INSTRUCTION = 'Write your final answer to the file `answer.txt` in the current working directory, in the form \\boxed{index}.'

def extract_answer(solution_text: str):
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, solution_text)
    if matches:
        return matches[-1].strip()
    return None

def apply_chat_template(toker, messages):
    input_prompt = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return toker(input_prompt, add_special_tokens=False).input_ids

def prepare_input_boxed(template, input_d):
    problem = input_d['problem']
    steps = input_d['steps']
    tagged_response = ''
    for sdx, step in enumerate(steps):
        tagged_response += f'<paragraph_{sdx}>\n{step}\n</paragraph_{sdx}>\n\n'
    tagged_response = tagged_response.strip()
    prompt = template.format(problem=problem, tagged_response=tagged_response)
    messages = [{'role': 'user', 'content': prompt}]
    return messages

def load_input_data(config, data_dir, ids):
    if data_dir is None:
        from datasets import load_dataset
        input_data = [e for e in load_dataset('Qwen/ProcessBench', split=config)]
    else:
        input_data = json.load(open(os.path.join(data_dir, f'{config}.json')))
    if ids is not None:
        input_data = [e for e in input_data if e['id'] in ids]
    return input_data

def run_codex(prompt, args, log_path):
    """Run `codex exec` on one prompt in a fresh working directory and return answer.txt."""
    with tempfile.TemporaryDirectory(prefix='processbench-codex-') as tmp:
        work_dir = os.path.join(tmp, 'work')
        codex_home = os.path.join(tmp, 'codex-home')
        os.makedirs(work_dir)
        os.makedirs(codex_home)
        if args.codex_config is not None:
            shutil.copy(args.codex_config, os.path.join(codex_home, 'config.toml'))
        # Same flags as Harbor's codex agent.
        cmd = [args.codex_bin, 'exec', '--dangerously-bypass-approvals-and-sandbox',
               '--skip-git-repo-check', '--model', args.model, '--json',
               '--enable', 'unified_exec']
        if args.reasoning_effort is not None:
            cmd += ['-c', f'model_reasoning_effort={args.reasoning_effort}']
        cmd += ['--', prompt]
        env = dict(os.environ, CODEX_HOME=codex_home)
        with open(log_path, 'w') as log:
            try:
                subprocess.run(cmd, cwd=work_dir, env=env, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT,
                               timeout=args.codex_timeout)
            except subprocess.TimeoutExpired:
                log.write(f'\n[run_eval] codex timed out after {args.codex_timeout}s\n')
        # Keep the session rollout (full trajectory), as Harbor does.
        sessions_dir = os.path.join(codex_home, 'sessions')
        if os.path.isdir(sessions_dir):
            shutil.copytree(sessions_dir, log_path[:-len('.jsonl')] + '_sessions', dirs_exist_ok=True)
        answer_path = os.path.join(work_dir, 'answer.txt')
        if os.path.isfile(answer_path):
            return open(answer_path, errors='replace').read()
        return ''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                        choices=['gsm8k', 'math', 'olympiadbench', 'omnimath'])
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument("--output_dir", type=str, default='./outputs')
    parser.add_argument('--use_voting', action='store_true')
    parser.add_argument('--voting_n', type=int, default=8)
    parser.add_argument('--backend', type=str, default='vllm', choices=['vllm', 'codex'])
    parser.add_argument('--data_dir', type=str, default=None,
                        help='load <config>.json from this dir instead of the HF hub')
    parser.add_argument('--ids_file', type=str, default=None,
                        help='only evaluate the item ids listed in this file (one per line)')
    # codex backend
    parser.add_argument('--model', type=str, default=None, help='model name passed to codex')
    parser.add_argument('--codex_bin', type=str, default='codex')
    parser.add_argument('--codex_config', type=str, default=None,
                        help='config.toml copied into a fresh CODEX_HOME for every item')
    parser.add_argument('--reasoning_effort', type=str, default=None)
    parser.add_argument('--codex_timeout', type=float, default=1800)
    parser.add_argument('--n_concurrent', type=int, default=8)
    args = parser.parse_args()

    TEMPLATE = open('./templates/critique_template.txt').read().strip()
    ids = None
    if args.ids_file is not None:
        ids = set(l.strip() for l in open(args.ids_file) if l.strip())

    if args.backend == 'codex':
        if args.model is None or args.use_voting:
            parser.error('--backend codex requires --model and does not support --use_voting')
        args.model_name = f'codex_{args.model}'
        run_codex_eval(args, TEMPLATE, ids)
        return

    if args.model_path is None:
        parser.error('--model_path is required for --backend vllm')
    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    args.model_name = os.path.basename(args.model_path)

    toker = AutoTokenizer.from_pretrained(args.model_path)


    llm = LLM(
        model=args.model_path, tokenizer=args.model_path,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=torch.cuda.device_count(),
        enable_prefix_caching=True, swap_space=16,
        max_num_seqs=20,
    )
    if not args.use_voting:
        sampling_params = SamplingParams(temperature=0.,
                                         max_tokens=32768 if 'QwQ' in args.model_path else 8192, seed=42)
    else:
        if 'Qwen2.5-Math' in args.model_path: # to ensure normal generation of Qwen2.5-Math-7B/72B-Instruct
            sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, n=args.voting_n,
                                            max_tokens=32768 if 'QwQ' in args.model_path else 8192, seed=42)
        else:
            sampling_params = SamplingParams(temperature=1, top_p=0.9, n=args.voting_n,
                                            max_tokens=32768 if 'QwQ' in args.model_path else 8192, seed=42)


    if args.configs is None:
        args.configs = ['gsm8k', 'math', 'olympiadbench', 'omnimath']

    for config in args.configs:
        if not args.use_voting:
            output_dir = os.path.join(args.output_dir, args.model_name)
        else:
            output_dir = os.path.join(args.output_dir, f'{args.model_name}_voting')
        os.makedirs(output_dir, exist_ok=True)

        input_data = load_input_data(config, args.data_dir, ids)
        prompt_token_ids = [apply_chat_template(toker, prepare_input_boxed(TEMPLATE, e))
                            for e in input_data]

        generations = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)


        res_data = []
        for i in range(len(input_data)):
            d = input_data[i].copy()

            if not args.use_voting:
                generated_critique = generations[i].outputs[0].text
                pred = extract_answer(generated_critique)
                try:
                    pred = int(pred)
                except:
                    pred = None
            else:
                generated_critique = [ee.text for ee in generations[i].outputs]
                preds = [extract_answer(e) for e in generated_critique]
                preds = [e for e in preds if e is not None]
                if len(preds) == 0:
                    pred = None
                else:
                    pred = Counter(preds).most_common(1)[0][0]
                    try:
                        pred = int(pred)
                    except:
                        pred = None

            d['generated_critique'] = generated_critique
            d['prediction'] = pred
            d['match'] = (pred == d['label'])

            res_data.append(d)

        save_and_report(res_data, output_dir, config)


def save_and_report(res_data, output_dir, config):
    error_data = [e for e in res_data if e['label'] != -1]
    correct_data = [e for e in res_data if e['label'] == -1]

    with open(os.path.join(output_dir, f'{config}_error.jsonl'), 'w') as f:
        for e in error_data:
            f.write(json.dumps(e) + '\n')
    with open(os.path.join(output_dir, f'{config}_correct.jsonl'), 'w') as f:
        for e in correct_data:
            f.write(json.dumps(e) + '\n')

    acc1 = np.mean([e['match'] for e in error_data]) * 100
    acc2 = np.mean([e['match'] for e in correct_data]) * 100
    f1 = 2 * acc1 * acc2 / (acc1 + acc2)
    print(f'{config} error acc: {acc1:.1f}, correct acc: {acc2:.1f}, f1: {f1:.1f}')
    return f1


def run_codex_eval(args, template, ids):
    if args.configs is None:
        args.configs = ['gsm8k', 'math', 'olympiadbench', 'omnimath']
    output_dir = os.path.join(args.output_dir, args.model_name)
    log_dir = os.path.join(output_dir, 'codex_logs')
    item_dir = os.path.join(output_dir, 'codex_items')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(item_dir, exist_ok=True)

    def evaluate(d):
        # Finished items are cached so an interrupted run can be resumed.
        item_path = os.path.join(item_dir, f"{d['id']}.json")
        if os.path.isfile(item_path):
            return json.load(open(item_path))
        d = dict(d)
        # Byte-identical to the Harbor task's instruction.md (which ends with a newline).
        prompt = prepare_input_boxed(template, d)[0]['content'] + '\n\n' + CODEX_OUTPUT_INSTRUCTION + '\n'
        generated_critique = run_codex(prompt, args, os.path.join(log_dir, f"{d['id']}.jsonl"))
        pred = extract_answer(generated_critique)
        try:
            pred = int(pred)
        except:
            pred = None
        d['generated_critique'] = generated_critique
        d['prediction'] = pred
        d['match'] = (pred == d['label'])
        with open(item_path, 'w') as f:
            json.dump(d, f)
        print(f"{d['id']} label={d['label']} prediction={pred} match={d['match']}", flush=True)
        return d

    # One pool across all configs so workers stay busy between subsets.
    input_data = {config: load_input_data(config, args.data_dir, ids) for config in args.configs}
    with ThreadPoolExecutor(max_workers=args.n_concurrent) as pool:
        futures = {config: [pool.submit(evaluate, d) for d in data]
                   for config, data in input_data.items()}
        res_data = {config: [f.result() for f in fs] for config, fs in futures.items()}

    f1s = [save_and_report(res_data[config], output_dir, config) for config in args.configs]
    print(f'average f1: {np.mean(f1s):.1f}')


if __name__ == '__main__':
    main()
