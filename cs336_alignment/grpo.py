import torch
from einops import rearrange
from typing import Callable
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer(prompt_strs, add_special_tokens=False, padding=False, truncation=True)["input_ids"]
    output_ids = tokenizer(output_strs, add_special_tokens=False, padding=False, truncation=True)["input_ids"]

    assert len(prompt_ids) == len(output_ids)

    input_ids = [prompt + response for prompt, response in zip(prompt_ids, output_ids)]

    # Pad the sequences
    max_len = max(len(ids) for ids in input_ids)
    padded_ids = []
    for id in input_ids:
        pad_len = max_len - len(id)
        padded_ids.append(id + [tokenizer.pad_token_id] * pad_len)

    input_ids = torch.tensor(padded_ids)

    # Construct mask
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for i, (prompt, output) in enumerate(zip(prompt_ids, output_ids)):
        len_prompt = len(prompt)
        len_output = len(output)
        response_mask[i, len_prompt : len_prompt + len_output] = 1


    labels = input_ids[:, 1:]
    input_ids = input_ids[:, :-1]
    response_mask = response_mask[:, 1:]

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    
    logits = model(input_ids).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1) # shape (batch_sz, seq_len, vocab_size)
    token_log_probs = rearrange(
        torch.gather(log_probs, 2, rearrange(labels, "b s -> b s 1")),
        "b s 1 -> b s",
    )

    res = {'log_probs': token_log_probs}
    if return_token_entropy:
        token_entropy = -torch.sum(log_probs * torch.exp(log_probs), dim=-1)
        res['token_entropy'] = token_entropy

    return res


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    assert len(rollout_responses) == len(repeated_ground_truths)

    all_rewards = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        # reward_dict contains keys 'format_reward', 'answer_reward', and 'reward'
        reward_dict = reward_fn(response, ground_truth)
        all_rewards.append(reward_dict)

    raw_rewards = torch.stack([torch.tensor(reward_dict['reward']) for reward_dict in all_rewards])

    mean_reward = sum(reward['reward'] for reward in all_rewards) / len(all_rewards)
    mean_format_reward = sum(reward['format_reward'] for reward in all_rewards) / len(all_rewards)
    metadata = {
        'mean_reward': mean_reward,
        'mean_format_reward': mean_format_reward,
    }

    return raw_rewards, metadata

