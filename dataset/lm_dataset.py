from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用 HuggingFace datasets 的惰性加载，避免一次性读入大文件
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    """
    我们拿到的是jsonl格式的每一行, 
    tokenizer会把文本转成input_id序列, 然后前后加上BOS/EOS标记, 最后右侧PAD到max_length
    然后编写lables,防止PAD参与loss计算,然后编写attention_mask, 让模型在计算注意力时忽略PAD位置
    我们要输出的是[input_ids, labels, attention_mask]
    """
    def __getitem__(self, index):
        sample = self.samples[index]

        # Step 1：tokenize 原始文本，留出首尾各 1 个 token 的位置给 BOS/EOS
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False, # 我们自己手动添加 BOS/EOS，所以关闭 tokenizer 的自动添加
            max_length=self.max_length - 2,  # 预留 BOS + EOS 的位置
            truncation=True, # 超过 max_length - 2 的部分会被截断，保证后续添加 BOS/EOS 后不超过 max_length
        ).input_ids

        # Step 2：拼接 BOS + token序列 + EOS，构成完整序列
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # Step 3：右侧用 PAD 补齐到 max_length，保证 batch 内等长
        input_ids = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens)
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Step 4：labels 与 input_ids 完全相同，但 PAD 位置置 -100，
        #         CrossEntropyLoss 会自动忽略 -100，不计入 loss
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # ！修正：返回 attention_mask，使 attention 层能屏蔽 padding token
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return input_ids, labels, attention_mask