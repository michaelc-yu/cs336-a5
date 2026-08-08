import torch
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
