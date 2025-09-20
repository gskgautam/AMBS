import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, logging
from huggingface_hub import login

@torch.no_grad
def gen_states(text, layer, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt")
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states[layer][0]
    return hidden_states

def list_to_str(lst):
    return ''.join(lst)

@torch.no_grad
def get_pos_neg_states(prompt_a, answer, prompt_b, layer, model, tokenizer, device):
    # prompt_a: Q1: question
    # answer: answer for q1
    # prompt_b: Q2 and Q3

    # generate 2 positives. 1st positive:
    # Q1: question\n
    # Answer: Half answer
    # Q2
    # Q3\n\n
    # Rest half of Q1
    # Encourages model to complete the answer for Q1 so that
    # it's easier to work when splitting into 3 copies

    # 2nd positive
    # Q1
    # Q2
    # Q3\n\n
    # Complete answer of Q1
    # Encourages model to provide helpful answers for the given question

    # negative:
    # Q1
    # Q2
    # Q3\n\n
    # EOS
    # discourages model to not provide blank answers
    
    tokenized_text = tokenizer.tokenize(answer)[1:]
    num_q1_tokens = len(tokenizer.tokenize(prompt_a))

    # </s> is EOS. Used here to align the negative and positive tokens
    negative_prompt = prompt_a + prompt_b + '</s>' * (len(tokenized_text) + 1)
    
    half_ans_len = len(tokenized_text) // 2
    half_ans_prompt = prompt_a + list_to_str(tokenized_text[:half_ans_len]) +\
                      prompt_b + list_to_str(tokenized_text[half_ans_len:]) + '</s>'
    half_ans_prompt_len = len(tokenizer.tokenize(prompt_a + list_to_str(tokenized_text[:half_ans_len])))
    pos2_prompt = prompt_a + prompt_b + answer + '</s>'

    # generate and align the hidden states
    # alignment is required to find the difference
    # no need for difference between positives and negatives for Q1 since
    # LLaMa is a causal LM and it wouldn't have any difference
    hidden_neg = gen_states(negative_prompt, layer, model, tokenizer, device)[num_q1_tokens:]
    hidden_half_ans = gen_states(half_ans_prompt, layer, model, tokenizer, device)[half_ans_prompt_len:]
    hidden_ans = gen_states(pos2_prompt, layer, model, tokenizer, device)[num_q1_tokens:]

    l = min(hidden_half_ans.shape[0], hidden_neg.shape[0])
    diff_a = (hidden_half_ans[:l, :] - hidden_neg[:l, :]).mean(dim = 0)
    diff_b = (hidden_ans - hidden_neg).mean(dim = 0)
    
    return diff_a, diff_b

# generate indices for 3 datasets having different number to samples to
# align. Generates simple repeating non-random lists
def generate_lists(a, b, c):
    a = int(a)
    b = int(b)
    c = int(c)
    nums = [a, b, c]
    max_val = max(nums)

    lists = []
    for n in nums:
        cycle = list(range(n))
        repeated = (cycle * ((max_val // len(cycle)) + 1))[:max_val]
        lists.append(repeated)
    return lists

# generates prompts from a row of alpaca, beavers, and tqa and uses them
# to generate hidden layer states
def get_states(alpaca, beavers, tqa, model, tokenizer, device):
    alpaca_prompt = alpaca['instruction']
    if alpaca['input']:
        alpaca_prompt += '\n' + alpaca['input']

    beavers_prompt = beavers['prompt']
    tqa_question = tqa['question']

    base_prompt_a = f'[INST]\nQ1: {alpaca_prompt}\nAnswer: '
    base_prompt_b = f'\n\nQ2: {beavers_prompt}\nAnswer: \n\nQ3: {tqa_question}\nAnswer: \n\n[/INST] '

    alpaca_output = alpaca['output']
    alpaca_diffs_a, alpaca_diffs_b = get_pos_neg_states(base_prompt_a,
                                                        alpaca_output,
                                                        base_prompt_b, model,
                                                        tokenizer, device)
    
    beavers_pos = beavers['positives']
    beavers_neg = beavers['negatives']

    if len(beavers_pos) == 0:
        beavers_pos = ['</s>' * 10]

    if len(beavers_neg) == 0:
        beavers_neg = ['</s>' * 10]

    bt_diffs_a = []
    base_prompt_a = f'[INST]Q: {beavers_prompt}\nAnswer: [/INST]'
    num_tokens = len(tokenizer.tokenize(base_prompt_a))

    pos_states = [gen_states(base_prompt_a + ' ' + str(pos) + '</s>',
                             model, tokenizer, device)[num_tokens:]
                             for pos in beavers_pos]

    neg_states = [gen_states(base_prompt_a + ' ' + str(neg) + '</s>',
                             model, tokenizer, device)[num_tokens:]
                             for neg in beavers_neg]

    for i in pos_states:
        for j in neg_states:
            l = min(i.shape[0], j.shape[0])
            diff = (i[:l, :] - j[:l, :]).mean(dim = 0)
            bt_diffs_a.append(diff)
    
    base_prompt_a = f'[INST]\nQ1: {alpaca_prompt}\nAnswer: \n\nQ2: {beavers_prompt}\nAnswer: '
    base_prompt_b = f'\n\nQ3: {tqa_question}\nAnswer: \n\n[/INST]'

    bt_diffs_b = []
    for pos in beavers_pos:
        for neg in beavers_neg:
            bt_diffs_b.append(get_pos_neg_states(base_prompt_a, str(pos),
                                                 str(neg), base_prompt_b,
                                                 model, tokenizer, device))

    tqa_pos = tqa['positives']
    tqa_neg = tqa['negatives']

    if len(tqa_pos) == 0:
        tqa_pos = ['</s>' * 10]

    if len(tqa_neg) == 0:
        tqa_neg = ['</s>' * 10]

    tqa_diffs_a = []
    base_prompt_a = f'[INST]Q: {tqa_question}\nAnswer: [/INST]'
    num_tokens = len(tokenizer.tokenize(base_prompt_a))

    pos_states = [gen_states(base_prompt_a + ' ' + str(pos) + '</s>',
                             model, tokenizer, device)[num_tokens:]
                             for pos in tqa_pos]

    neg_states = [gen_states(base_prompt_a + ' ' + str(neg) + '</s>',
                             model, tokenizer, device)[num_tokens:]
                             for neg in tqa_neg]

    for i in pos_states:
        for j in neg_states:
            l = min(i.shape[0], j.shape[0])
            diff = (i[:l, :] - j[:l, :]).mean(dim = 0)
            tqa_diffs_a.append(diff)

    base_prompt_a = f'[INST]Q1: {alpaca_prompt}\nAnswer: \n\nQ2: {beavers_prompt}\nAnswer: \n\nQ3: {tqa_question}\nAnswer: '
    base_prompt_b = f'\n\n[/INST]'

    tqa_diffs_b = []
    for pos in tqa_pos:
        for neg in tqa_neg:
            tqa_diffs_b.append(get_pos_neg_states(base_prompt_a, pos, neg,
                                                  base_prompt_b, model,
                                                  tokenizer, device))
    return alpaca_diffs_a, alpaca_diffs_b,\
           bt_diffs_a, bt_diffs_b,\
           tqa_diffs_a, tqa_diffs_b
    
def preprocess(alpaca_path, beavers_path, tqa_path, model_name, layer, device):
    alpaca_train = pd.read_json(alpaca_path)
    bt_train = pd.read_csv(beavers_path)
    tqa_train = pd.read_csv(tqa_path)

    # group by questions into 2 columns: positives and negatives for safe and unsafe responses, if any
    bt_train = (
        bt_train.groupby("prompt")
          .apply(lambda g: pd.Series({
              "positives": g.loc[g["is_safe"] == 1, "response"].tolist(),
              "negatives": g.loc[g["is_safe"] == 0, "response"].tolist()
          })).reset_index()
    )

    # group by questions into 2 columns: positives and negatives for true and untrue responses, if any
    tqa_train = (
        tqa_train.groupby("question")
          .apply(lambda g: pd.Series({
              "positives": g.loc[g["label"] == 1, "answer"].tolist(),
              "negatives": g.loc[g["label"] == 0, "answer"].tolist()
          })).reset_index()
    )

    # use less than available samples for faster training for alpaca
    num_samples_alpaca = 10000
    num_samples_beavers = len(bt_train)
    num_samples_tqa = len(tqa_train)
    login('huggingface token')
    logging.set_verbosity(logging.CRITICAL)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModel.from_pretrained(
            model_name,
            use_auth_token=True,
            device_map=device,
            torch_dtype='auto',
            trust_remote_code=True
        ).eval()

    train_indices = generate_lists(num_samples_alpaca, num_samples_beavers, num_samples_tqa)
    alpaca_diffs_a = alpaca_diffs_b = []
    bt_diffs_a = bt_diffs_b = []
    tqa_diffs_a = tqa_diffs_b = []

    for i in range(base_index, base_index + num_trains):
        diffs = get_states(alpaca_train.iloc[train_indices[0][i]],
                           bt_train.iloc[train_indices[1][i]],
                           tqa_train.iloc[train_indices[2][i]])
        alpaca_diffs_a.append(diffs[0])
        alpaca_diffs_b.append(diffs[1])
        bt_diffs_b.extend(diffs[2])
        bt_diffs_a.extend(diffs[3])
        tqa_diffs_a.extend(diffs[4])
        tqa_diffs_b.extend(diffs[5])

    return alpaca_diffs_a, alpaca_diffs_b,\
           bt_diffs_a, bt_diffs_b,\
           tqa_diffs_a, tqa_diffs_b