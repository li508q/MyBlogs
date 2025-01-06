---
title: Huggingface-PreTrain-01-MLM
description: 整理中
date: 2025-01-04 12:00:00
updated: 2025-01-04 12:00:00
cover: /img/huggingface.png
top_img: /img/huggingface_long.png
tags: Huggingface
categories: 框架
---

## 预训练

预训练(PreTrain)是从大规模数据中, 通过自监督学习的方式, 获得与具体任务无关的预训练模型(Pretrained Model)的过程. 模型通常为三类:
1. 编码器模型(自编码模型, 掩码语言模型MLM): BERT, 适用 `文本分类`, `命名实体识别`, `阅读理解` 任务
	- 只计算掩码部分的 loss
2. 解码器模型(自回归模型, 因果语言模型CausalLM): GPT, 适用 `文本生成` 任务
	- 计算 `第一个` 到 `倒数第二个` tokens 的 loss
3. 编码器解码器模型(Seq2Seq模型, 前缀语言模型PLM): BART, T5, 适用 `文本摘要`, `机器翻译` 任务
	- 只计算解码器部分的 Loss
	- 注意 PLM 模型结构其实是 单塔Decoder-Only + 特殊attention_mask
		- 任务形式很像 Seq2Seq
		- 在 attention_mask 部分, 前一部分 Seq 是双向的, 后一部分 Seq 是下三角

## MLM

本文预训练的任务是掩码语言模型(MLM), 自编码模型:
- 将一些位置的 tokens 替换成特殊字符 \[MASK\]
- 预测这些被替换的字符, 只计算掩码部分的 loss, 其余部分不计算 loss

## 实战

数据集: pleisto/wikipedia-cn-20230720-filtered
预训练模型: hfl/chinese-macbert-base

为避免网络问题, 在脚本开始位置配置镜像环境变量: 
```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

### 加载dataset

```python
from datasets import Dataset
ds_path = "../datasets/wiki_cn_filtered"
ds = Dataset.load_from_disk(ds_path)
ds = ds.select(range(10000))
ds
```

```
Dataset({
    features: ['completion', 'source'],
    num_rows: 10000
})
```

```python
ds[0]
```

```
{'completion': '昭通机场（ZPZT）是位于中国云南昭通的民用机场，始建于1935年，1960年3月开通往返航班“昆明－昭通”，原来属军民合用机场。1986年机场停止使用。1991年11月扩建，于1994年2月恢复通航。是西南地区「文明机场」，通航城市昆明。 机场占地1957亩，飞行区等级为4C，有一条跑道，长2720米，宽48米，可供波音737及以下机型起降。机坪面积6600平方米，停机位2个，航站楼面积1900平方米。位于城东6公里处，民航路与金鹰大道交叉处。\n航点\n客服电话\n昭通机场客服电话：0870-2830004',
 'source': 'wikipedia.zh2307'}
```

### 加载 tokenizer

```python
from transformers import AutoTokenizer

model_id_or_path = "../models/bloom-389m-zh"
tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
```

查看一下 tokenizer 的特殊字符及其 id
```python
tokenizer.pad_token, tokenizer.pad_token_id
```

```
('<pad>', 3)
```

```python
tokenizer.eos_token, tokenizer.eos_token_id
```

```
('</s>', 2)
```

### 对 DataSet 进行数据处理, 使用 map 方法

主要是实现文本的 tokenization, 同时在 CausalLM 下, 我们需要给句子添加 eos_token
```python
def tokenize_function(examples):
    content = [e + tokenizer.eos_token for e in examples["completion"]]
    return tokenizer(examples["completion"], max_length=384, truncation=True)

tokenized_ds = ds.map(tokenize_function, batched=True, remove_columns=ds.column_names)

tokenized_ds
```

```
Dataset({
    features: ['input_ids', 'attention_mask'],
    num_rows: 10000
})
```

### 构建Collator, 创建DataLoader

我们使用 `DataCollatorForLanguageModeling` 作为 `collator_fn`:
- `DataCollatorForLanguageModeling` 默认情况下 `mlm=True`
- `mlm_probability=0.15` 定义掩码比例
```python
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

dl = DataLoader(
		tokenized_ds, 
		batch_size=2, 
		collate_fn=DataCollatorForLanguageModeling(
			tokenizer, 
			mlm=True, 
			mlm_probability=0.15
	), 
		shuffle=True
)
```

```python
next(iter(dl))
```
其输出的字典, keys有:
- `input_ids`
	- 原始文本转成的 token
	- 被掩部分 token 替换为 mask_token_id
	- padding 部分为 pad_token_id

- `token_type_ids`
	- 本文加载 tokenizer 并未将其定义为 BERT 类, 不会使用 `token_type_ids`, 也可能直接生成一串全 0 或者干脆不输出这个字段
	- 如果 tokenizer 是 BERT 类, 生成 `token_type_ids`以供 BERT 这类双向模型使用. 对于单句子数据, 通常全部都为 0; 如有分割符 `[SEP]` 并拼接多句子输入, 第二句则会是 1

- `attention_mask`
	- 标记 padding

- `labels`
	- mask 部分为 mask 前的实际 token 作为 label
	- 未被 mask 部分为 -100

我们可以查看一下 mask_token: 
```python
tokenizer.mask_token, tokenizer.mask_token_id
```

```
('[MASK]', 103)
```

### 加载模型, 配置训练参数和 Trainer

```python
from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained(model_id_or_path)

from transformers import Trainer, TrainingArguments
args = TrainingArguments(
    output_dir="../output/mlm_chinese-macbert-base",
    overwrite_output_dir=True,
    per_device_train_batch_size=64,
    logging_steps=10,
    num_train_epochs=1,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForLanguageModeling(
	    tokenizer, 
	    mlm=True, 
	    mlm_probability=0.15
	),
)
```

因为我们用 AutoModelForMaskedLM 类去加载一个 BERT 类的模型, 因为可能会输出提醒, 部分权重并未被加载, 这是正常的, BERT 还存在分类头等部分架构, 这部分不会被 MLM 类加载.

### 训练

```python
trainer.train()
```

```
TrainOutput(global_step=157, training_loss=1.318490326024924, metrics={'train_runtime': 89.1646, 'train_samples_per_second': 112.152, 'train_steps_per_second': 1.761, 'total_flos': 1973819658240000.0, 'train_loss': 1.318490326024924, 'epoch': 1.0})
```

### 推理

```python
from transformers import pipeline

fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
fill_mask("今天[MASK]情很好")
```

```
['Device set to use cuda:0\n']
[{'score': 0.7547478675842285,
  'token': 2552,
  'token_str': '心',
  'sequence': '今 天 心 情 很 好'},
 {'score': 0.0374448262155056,
  'token': 2697,
  'token_str': '感',
  'sequence': '今 天 感 情 很 好'},
 {'score': 0.031073328107595444,
  'token': 6121,
  'token_str': '行',
  'sequence': '今 天 行 情 很 好'},
 {'score': 0.01597883366048336,
  'token': 722,
  'token_str': '之',
  'sequence': '今 天 之 情 很 好'},
 {'score': 0.015516223385930061,
  'token': 4638,
  'token_str': '的',
  'sequence': '今 天 的 情 很 好'}]
```
