---
title: Huggingface-PreTrain-02-CausalLM(GPT)
description: 因果语言模型CausalLM的训练流程
date: 2025-01-04 20:00:00
updated: 2025-01-04 20:00:00
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

## CausalLM

本文预训练的任务是因果语言模型(CasualLM), 自回归模型:
- 输入完整序列
- 基于上文 tokens 预测当前 token
- 结束位置需要有特殊字符 eos_token, 使语言学习到生成结束的信息

## 流程细节

数据集: pleisto/wikipedia-cn-20230720-filtered
预训练模型: Langboat/bloom-389m-zh

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

主要是实现文本的 tokenization
- 注意在 CausalLM 下, 我们需要给句子添加 eos_token, 从而帮助模型学会在生成时适时停止

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

**有关 bos_token 的讨论:**

bos_token 通常在解码或生成阶段使用, 有些框架或应用会把 bos_token 放在开头, 显式告诉模型“从这里开始生成”: 

- 注意我们这里并没有拼接 bos_token, 这是由于我们预训练的 `GPT系列` 是 `Decoder-Only架构`, 模型往往会直接在序列开头开始训练, 因而 bos_token 在 GPT 预训练阶段并不必需.
	- 如果需要指示新的上下文开头, 可以通过其他方式
	- 如拼接新文档时插入 eos_token 做分隔, 或用特定的分隔符

- 通常视后续下游任务确认是否会使用 bos_token
	- 对于对话模型，有时也会在新一轮对话的开头加上 `<BOS>` 以区分上一轮对话末尾和下一轮开始
	- 但有时对于对话会添加额外的系统消息或角色定义, 如 `<|system|>`, `<|user|>`, `<|assistant|>`, 并且上下文能够清晰地分段, 只要模型能正确区分 对话 / 指令 的上下文, 就不一定要再塞一个 bos_token

### 构建Collator, 创建DataLoader

注意此时的 `DataCollatorForLanguageModeling` 需要传入 `mlm=False`, 以保证不进行 `掩码操作`:

```python
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

dl = DataLoader(tokenized_ds, batch_size=1, collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False), shuffle=True)
```

> 这里有一个值得注意的地方:
> 
> 我们知道对于 CausalLM 而言, 当前 token 的label 就是下一个 token
> 
> 但是此时 dl 所输出的 labels 还是 input 自身, 我们可以如下验证

```python
tmp = next(iter(dl))
tmp['labels'] == tmp['input_ids']
```

```
tensor([[True, ..., True]])
```


> 至于 labels 的 shift 操作是在实际模型中进行的, 在每个模型中有关 CausalLM 的类中的 forward 部分,  loss 计算会进行处理

### 加载模型, 配置训练参数和 Trainer

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model_id_or_path)
```

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="../output/gpt_bloom_389m_zh",
    overwrite_output_dir=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=1,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
```

### 训练

```python
trainer.train()
```

```
TrainOutput(global_step=156, training_loss=3.615333398183187, metrics={'train_runtime': 398.9013, 'train_samples_per_second': 25.069, 'train_steps_per_second': 0.391, 'total_flos': 6954157911048192.0, 'train_loss': 3.615333398183187, 'epoch': 0.9984})
```

### 推理

查看一下训练的效果, 注意此时刚结束训练, GPU 显存占用可能过高, 可以清除一下 cache

```python
import torch
torch.cuda.empty_cache()
```

```python
from transformers import pipeline

generator = pipeline('text-generation', model=model, tokenizer=tokenizer, do_sample=True)

generator("下面是一则娱乐新闻,", max_length=50)
```