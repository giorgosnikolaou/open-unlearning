from torch.utils.data import Dataset

from data.utils import load_hf_dataset, add_dataset_index, preprocess_chat_instance


class RWKUDPODataset(Dataset):
    """Dataset for RWKU DPO paired training data.

    Handles RWKU ``train_pair_*`` configs where ``response`` is an array of 2
    strings: [positive/counterfactual, negative/factual]. Returns items in the
    ``{"original": ..., "alternate": ...}`` format expected by the DPO trainer.

    ``original`` = factual response (to forget / losing in DPO)
    ``alternate`` = counterfactual response (to keep / winning in DPO)
    """

    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        prompt_key="prompt",
        response_key="response",
        intro_key="intro",
        max_length=512,
        predict_with_generate=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_args = template_args
        self.predict_with_generate = predict_with_generate

        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)

        self.prompt_key = prompt_key
        self.response_key = response_key
        self.intro_key = intro_key

    def __len__(self):
        return len(self.data)

    def _process_sample(self, question, answer, index=-1):
        tokenized = preprocess_chat_instance(
            self.tokenizer,
            self.template_args,
            [question],
            [answer],
            self.max_length,
            self.predict_with_generate,
        )
        item = {
            "input_ids": tokenized["input_ids"],
            "labels": tokenized["labels"],
            "attention_mask": tokenized["attention_mask"],
            "index": index,
        }
        return item

    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt = sample[self.prompt_key]
        responses = sample[self.response_key]
        index = sample["index"]

        # responses[0] = positive/counterfactual (winning in DPO)
        # responses[1] = negative/factual (losing in DPO — the knowledge to forget)
        original = self._process_sample(prompt, responses[1], index)
        alternate = self._process_sample(prompt, responses[0], index)

        return {"original": original, "alternate": alternate}
