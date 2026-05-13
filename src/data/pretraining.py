# import torch
import random
from collections import defaultdict

from torch.utils.data import Dataset
from tqdm.auto import tqdm

from data.utils import (
    load_hf_dataset,
    add_dataset_index,
    preprocess_pretraining_instance,
    preprocess_chat_instance,
)


class CompletionDataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        prefix_key="prompt",
        text_key="text",
        max_length=2048,
        predict_with_generate=False,
        insert_space=False,
    ):
        super(CompletionDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        # if either key does not exist in dataset, it is taken as ""
        self.prefix_key = prefix_key
        self.text_key = text_key
        self.predict_with_generate = predict_with_generate
        self.insert_space = insert_space

    def __len__(self):
        return len(self.data)

    def _process_sample(self, prefix, text_content, index=-1):
        tokenized_data = preprocess_pretraining_instance(
            self.tokenizer,
            prefix,
            text_content,
            self.max_length,
            self.predict_with_generate,
            self.insert_space,
        )
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
        }
        if index != -1:
            item_dct["index"] = index
        return item_dct

    def __getitem__(self, idx):
        pref = self.data[idx].get(self.prefix_key, "")
        text_content = self.data[idx].get(self.text_key, "")
        index = self.data[idx]["index"]
        item = self._process_sample(pref, text_content, index)
        return item


class PretrainingDataset(Dataset):
    def __init__(
        self, hf_args, template_args, tokenizer, text_key="text", max_length=2048
    ):
        super(PretrainingDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunks = self._chunk_raw_text(load_hf_dataset(**hf_args)[text_key])

    def _chunk_raw_text(self, raw_text):
        raw_text = "\n\n".join(raw_text)
        full_token_sequence = self.tokenizer(raw_text, add_special_tokens=False)[
            "input_ids"
        ]
        num_chunks = len(full_token_sequence) // self.max_length + 1
        chunks = []
        for i in range(num_chunks):
            chunks.append(
                self.tokenizer.decode(
                    full_token_sequence[i * self.max_length : (i + 1) * self.max_length]
                )
            )
        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return preprocess_pretraining_instance(
            self.tokenizer, "", self.chunks[idx], self.max_length
        )


class PromptedCompletionDataset(Dataset):
    """CompletionDataset that wraps each sample in a chat template with a
    per-sample system prompt suffix (e.g. 'Give all information about {subject}').

    The text becomes the assistant's response. A configurable user prompt
    (also supporting {subject} interpolation) is used as the user turn.
    """

    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        text_key="text",
        subject_key="subject",
        user_prompt="Tell me about {subject}.",
        # system_prompt_suffix="Give information about {subject}.",
        system_prompt_suffix="Give information related only to the requested subject.",
        max_length=512,
        predict_with_generate=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        self.base_template_args = dict(template_args)
        self.text_key = text_key
        self.subject_key = subject_key
        self.user_prompt_template = user_prompt
        self.suffix_template = system_prompt_suffix
        self.predict_with_generate = predict_with_generate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample[self.text_key]
        subject = sample[self.subject_key]
        index = sample["index"]

        # Build per-sample template_args with dynamic suffix
        template_args = dict(self.base_template_args)
        base_prompt = template_args.get("system_prompt", "")
        suffix = self.suffix_template.format(subject=subject)
        if base_prompt:
            template_args["system_prompt"] = base_prompt + " " + suffix

        user_msg = self.user_prompt_template.format(subject=subject)

        # Truncate text to max_length tokens
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(text_ids) > self.max_length:
            text_ids = text_ids[: self.max_length]
            text = self.tokenizer.decode(text_ids)

        tokenized = preprocess_chat_instance(
            self.tokenizer,
            template_args,
            [user_msg],
            [text],
            self.max_length,
            self.predict_with_generate,
        )
        
        # print(self.tokenizer.decode(tokenized['input_ids']))
        # exit(0)

        return {
            "input_ids": tokenized["input_ids"],
            "labels": tokenized["labels"],
            "attention_mask": tokenized["attention_mask"],
            "index": index,
        }


class PartitionedPretrainingDataset(Dataset):
    """Like PretrainingDataset but chunks within each partition (e.g., subject).

    Groups samples by ``partition_key``, concatenates texts within each group
    with ``\\n\\n``, and chunks into fixed-length blocks.  This avoids mixing
    content across partitions in a single chunk.

    When ``per_sample=True``, skips concatenation and chunking — each original
    sample is kept as an independent training example (truncated to
    ``max_length``).  This matches per-sample processing used by other
    frameworks like LLaMA-Factory.
    """

    def __init__(
        self, hf_args, template_args, tokenizer, text_key="text",
        partition_key="subject", max_length=512, random_crop=False,
        per_sample=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.random_crop = random_crop
        self.per_sample = per_sample

        data = load_hf_dataset(**hf_args)

        if per_sample:
            self.samples = [sample[text_key] for sample in data]
        else:
            groups = defaultdict(list)
            for sample in data:
                groups[sample[partition_key]].append(sample[text_key])

            if random_crop:
                self.partitions = []
                self.index_map = []
                for texts in groups.values():
                    raw_text = "\n\n".join(texts)
                    tokens = tokenizer(raw_text, add_special_tokens=False)["input_ids"]
                    part_idx = len(self.partitions)
                    self.partitions.append(tokens)
                    num_chunks = max(1, len(tokens) // max_length + 1)
                    self.index_map.extend([part_idx] * num_chunks)
            else:
                self.chunks = []
                for texts in tqdm(groups.values(), desc='Chunking Dataset'):
                    self.chunks.extend(self._chunk_raw_text(texts))

    def _chunk_raw_text(self, raw_texts):
        raw_text = "\n\n".join(raw_texts)
        full_token_sequence = self.tokenizer(raw_text, add_special_tokens=False)[
            "input_ids"
        ]
        num_chunks = len(full_token_sequence) // self.max_length + 1
        chunks = []
        for i in range(num_chunks):
            chunks.append(
                self.tokenizer.decode(
                    full_token_sequence[i * self.max_length : (i + 1) * self.max_length]
                )
            )
        return chunks

    def __len__(self):
        if self.per_sample:
            return len(self.samples)
        if self.random_crop:
            return len(self.index_map)
        return len(self.chunks)

    def __getitem__(self, idx):
        if self.per_sample:
            return preprocess_pretraining_instance(
                self.tokenizer, "", self.samples[idx], self.max_length
            )
        if self.random_crop:
            tokens = self.partitions[self.index_map[idx]]
            if len(tokens) <= self.max_length:
                start = 0
            else:
                start = random.randint(0, len(tokens) - self.max_length)
            crop = tokens[start : start + self.max_length]
            text = self.tokenizer.decode(crop)
            return preprocess_pretraining_instance(
                self.tokenizer, "", text, self.max_length
            )
        return preprocess_pretraining_instance(
            self.tokenizer, "", self.chunks[idx], self.max_length
        )


class PartitionedPromptedDataset(Dataset):
    """Hybrid of PartitionedPretrainingDataset and PromptedCompletionDataset.

    Concatenates all passages per subject, chunks them into sequential
    ``max_length``-token blocks, and wraps each chunk in a chat template
    with a per-subject system prompt and user prompt.
    """

    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        text_key="text",
        partition_key="subject",
        user_prompt="Tell me about {subject}.",
        system_prompt_suffix="Give information related only to the requested subject.",
        max_length=512,
        predict_with_generate=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.base_template_args = dict(template_args)
        self.user_prompt_template = user_prompt
        self.suffix_template = system_prompt_suffix
        self.predict_with_generate = predict_with_generate

        data = load_hf_dataset(**hf_args)

        groups = defaultdict(list)
        for sample in data:
            groups[sample[partition_key]].append(sample[text_key])

        self.chunks = []  # list of (subject_name, chunk_text)
        for subject, texts in groups.items():
            raw_text = "\n\n".join(texts)
            token_ids = tokenizer(raw_text, add_special_tokens=False)["input_ids"]
            num_chunks = max(1, len(token_ids) // max_length + 1)
            for i in range(num_chunks):
                chunk_ids = token_ids[i * max_length : (i + 1) * max_length]
                self.chunks.append((subject, tokenizer.decode(chunk_ids)))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        subject, text = self.chunks[idx]

        # Build chat template
        template_args = dict(self.base_template_args)
        base_prompt = template_args.get("system_prompt", "")
        suffix = self.suffix_template.format(subject=subject)
        if base_prompt:
            template_args["system_prompt"] = base_prompt + " " + suffix

        user_msg = self.user_prompt_template.format(subject=subject)

        tokenized = preprocess_chat_instance(
            self.tokenizer,
            template_args,
            [user_msg],
            [text],
            self.max_length,
            self.predict_with_generate,
        )

        return {
            "input_ids": tokenized["input_ids"],
            "labels": tokenized["labels"],
            "attention_mask": tokenized["attention_mask"],
        }
