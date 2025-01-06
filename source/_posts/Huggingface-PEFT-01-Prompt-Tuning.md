---
title: Huggingface-PEFT-01-Prompt-Tuning
description: 整理中
date: 2025-01-05 12:00:00
updated: 2025-01-05 12:00:00
cover: /img/huggingface.png
top_img: /img/huggingface_long.png
tags: Huggingface
categories: 框架
---

## 什么是 Prompt-Tuning ?

1. 冻结主模型全部参数
2. 训练数据前加入一小段 Prompt, 只训练这一小段的 Prompt 表示
    - 也就是一个 Embedding 模块
    - 两种形式: Hard Prompt 和 Soft Prompt

## 处理数据与加载模型

数据集: shibing624/alpaca-zh
预训练模型: Langboat/bloom-1b4-zh

为避免网络问题, 在脚本开始位置配置镜像环境变量: 
```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

### 加载数据

```python
from datasets import load_from_disk
ds_path = "../datasets/shibing624/alpaca-zh"
ds = load_from_disk(ds_path)['train']
ds = ds.select(range(10000))
ds

```

```
Dataset({
    features: ['instruction', 'input', 'output'],
    num_rows: 10000
})
```

```python
ds[:3]
```

```
{'instruction': ['保持健康的三个提示。', '三原色是什么？', '描述原子的结构。'],
 'input': ['', '', ''],
 'output': ['以下是保持健康的三个提示：\n\n1. 保持身体活动。每天做适当的身体运动，如散步、跑步或游泳，能促进心血管健康，增强肌肉力量，并有助于减少体重。\n\n2. 均衡饮食。每天食用新鲜的蔬菜、水果、全谷物和脂肪含量低的蛋白质食物，避免高糖、高脂肪和加工食品，以保持健康的饮食习惯。\n\n3. 睡眠充足。睡眠对人体健康至关重要，成年人每天应保证 7-8 小时的睡眠。良好的睡眠有助于减轻压力，促进身体恢复，并提高注意力和记忆力。',
  '三原色通常指的是红色、绿色和蓝色（RGB）。它们是通过加色混合原理创建色彩的三种基础颜色。在以发光为基础的显示设备中（如电视、计算机显示器、智能手机和平板电脑显示屏）, 三原色可混合产生大量色彩。其中红色和绿色可以混合生成黄色，红色和蓝色可以混合生成品红色，蓝色和绿色可以混合生成青色。当红色、绿色和蓝色按相等比例混合时，可以产生白色或灰色。\n\n此外，在印刷和绘画中，三原色指的是以颜料为基础的红、黄和蓝颜色（RYB）。这三种颜色用以通过减色混合原理来创建色彩。不过，三原色的具体定义并不唯一，不同的颜色系统可能会采用不同的三原色。',
  '原子是物质的基本单位，它由三种基本粒子组成：质子、中子和电子。质子和中子形成原子核，位于原子中心，核外的电子围绕着原子核运动。\n\n原子结构具有层次性。原子核中，质子带正电，中子不带电（中性）。原子核非常小且致密，占据了原子总质量的绝大部分。电子带负电，通常围绕核运动，形成若干层次，称为壳层或电子层。电子数量与质子数量相等，使原子呈电中性。\n\n电子在每个壳层中都呈规律分布，并且不同壳层所能容纳的电子数也不同。在最里面的壳层一般只能容纳2个电子，其次一层最多可容纳8个电子，再往外的壳层可容纳的电子数逐层递增。\n\n原子核主要受到两种相互作用力的影响：强力和电磁力。强力的作用范围非常小，主要限制在原子核内，具有极强的吸引作用，使核子（质子和中子）紧密结合在一起。电磁力的作用范围较大，主要通过核外的电子与原子核相互作用，发挥作用。\n\n这就是原子的基本结构。原子内部结构复杂多样，不同元素的原子核中质子、中子数量不同，核外电子排布分布也不同，形成了丰富多彩的化学世界。']}
```

### 数据处理

```python
from transformers import AutoTokenizer

model_id_or_path = "../models/bloom-1b4-zh"
tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)

# 定义 map方法
def process_function(examples):
    MAX_LEN = 256
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer("\n".join(["Human: " + examples['instruction'], examples['input']]).strip() + "\n\nAssistant: ")
    response = tokenizer(examples['output'] + tokenizer.eos_token)
    input_ids = instruction['input_ids'] + response['input_ids']
    attention_mask = instruction['attention_mask'] + response['attention_mask']
    labels = [-100] * len(instruction['input_ids']) + response['input_ids']
    if len(input_ids) > MAX_LEN:
        input_ids = input_ids[:MAX_LEN]
        attention_mask = attention_mask[:MAX_LEN]
        labels = labels[:MAX_LEN]
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

tokenized_ds = ds.map(process_function, remove_columns=ds.column_names)
tokenized_ds
```

```
Dataset({
    features: ['input_ids', 'attention_mask', 'labels'],
    num_rows: 10000
})
```

**检查一下 input 内容 和 label 内容的对应:**

```python

tokenizer.decode(tokenized_ds[0]['input_ids'])

```

```
'Human: 保持健康的三个提示。\n\nAssistant: 以下是保持健康的三个提示：\n\n1. 保持身体活动。每天做适当的身体运动，如散步、跑步或游泳，能促进心血管健康，增强肌肉力量，并有助于减少体重。\n\n2. 均衡饮食。每天食用新鲜的蔬菜、水果、全谷物和脂肪含量低的蛋白质食物，避免高糖、高脂肪和加工食品，以保持健康的饮食习惯。\n\n3. 睡眠充足。睡眠对人体健康至关重要，成年人每天应保证 7-8 小时的睡眠。良好的睡眠有助于减轻压力，促进身体恢复，并提高注意力和记忆力。</s>'
```

```python
tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[0]['labels'])))
```

```
'以下是保持健康的三个提示：\n\n1. 保持身体活动。每天做适当的身体运动，如散步、跑步或游泳，能促进心血管健康，增强肌肉力量，并有助于减少体重。\n\n2. 均衡饮食。每天食用新鲜的蔬菜、水果、全谷物和脂肪含量低的蛋白质食物，避免高糖、高脂肪和加工食品，以保持健康的饮食习惯。\n\n3. 睡眠充足。睡眠对人体健康至关重要，成年人每天应保证 7-8 小时的睡眠。良好的睡眠有助于减轻压力，促进身体恢复，并提高注意力和记忆力。</s>'
```

### 探索 Collator

这里我们需要多探索一下 Collator 的选用, 因为后续配置 Trainer 的时候会发现, 我们使用的是 `DataCollatorForSeq2Seq` 而非在预训练的时候所使用 `DataCollatorForLanguageModeling`.

```python
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from transformers import DataCollatorForSeq2Seq

dl_FLM = DataLoader(tokenized_ds, batch_size=1, collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False))
dl_FS2S = DataLoader(tokenized_ds, batch_size=1, collate_fn=DataCollatorForSeq2Seq(tokenizer, padding=True))

test_dl_FLM = next(iter(dl_FLM))
test_dl_FS2S = next(iter(dl_FS2S))
```

比较`test_dl_FLM` 和 `test_dl_FS2S` :
- 结构是一致的: `dict_keys(['input_ids', 'attention_mask', 'labels'])`
- `input_ids`, `attention_mask` 内容一致
- `labels`出现了不一致
	- `DataCollatorForLanguageModeling` 在处理时秉持的是 `自回归`的逻辑, 所以他会抛弃掉我们数据集中已有的 `labels`, 而是基于 `input_ids` 生成与之一致的 `labels`
	- `DataCollatorForSeq2Seq` 会保持数据集中原有的 `labels`

```python
dl_FLM = DataLoader(tokenized_ds, batch_size=2, collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False))
dl_FS2S = DataLoader(tokenized_ds, batch_size=2, collate_fn=DataCollatorForSeq2Seq(tokenizer, padding=True))
```
将 `batch_size` 设大:
- `DataCollatorForLanguageModeling` 会出现冲突问题: 
	- ValueError: Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your features (`labels` in this case) have excessive nesting (inputs type `list` where type `int` is expected).
	- 因为我们在预训练使用 `DataCollatorForLanguageModeling` 时的 `Dataset`结构是
		- `Dataset({ features: ['input_ids', 'attention_mask'], num_rows: 10000 })`
		- 此时: `Dataset({ features: ['input_ids', 'attention_mask', 'labels'], num_rows: 10000 })`, 多了一个字段
			- 对 `labels` 删除(`del_col = tokenized_ds.remove_columns(["labels"])`): 有效, 但原有 `labels` 被破坏 
			- 设想是不是 `DataCollatorForLanguageModeling` 处理时会把 `labels` 视作已经处理好的 `labels`. 因而, 对 `labels` 更名(`renamed = tokenized_ds.rename_column("labels", "original")`): 无效
		- 故而, 目前结论是 `DataCollatorForLanguageModeling` 处理时需要数据集结构保持 `Dataset({ features: ['input_ids', 'attention_mask'], num_rows: 10000 })`, **具体原因和细节还需要后续解读源码**

### 加载模型

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_id_or_path)
model

```

**输出一下模型架构, 供后续与 prompt_model 进行比较**:

```
BloomForCausalLM(
  (transformer): BloomModel(
    (word_embeddings): Embedding(46145, 2048)
    (word_embeddings_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
    (h): ModuleList(
      (0-23): 24 x BloomBlock(
        (input_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (self_attention): BloomAttention(
          (query_key_value): Linear(in_features=2048, out_features=6144, bias=True)
          (dense): Linear(in_features=2048, out_features=2048, bias=True)
          (attention_dropout): Dropout(p=0.0, inplace=False)
        )
        (post_attention_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (mlp): BloomMLP(
          (dense_h_to_4h): Linear(in_features=2048, out_features=8192, bias=True)
          (gelu_impl): BloomGelu()
          (dense_4h_to_h): Linear(in_features=8192, out_features=2048, bias=True)
        )
      )
    )
    (ln_f): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=2048, out_features=46145, bias=False)
)
```

计算一下参数量: 

```python
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

```
1303111680
```

模型存储为 FP32: 
- Model Size: 1.3B
- model: 1.3B * 4 
- gradient: 1.3B * 4
- optimizer: 1.3B * 4 * 2
- sum: ~= 20.8G

## Prompt Tuning

Huggingface 提供了 peft 库, 通过这一个库我们可以实现 PEFT Model 的加载.
### Soft Prompt-Tuning v.s. Hard Prompt-Tuning

首先我们需要先进行 `PromptTuningConfig` 的配置:

```python
from peft import PromptTuningConfig, TaskType, PromptTuningInit

## Soft Prompt Tuning
# config = PromptTuningConfig(
#	  task_type=TaskType.CAUSAL_LM, 
#	  num_virtual_tokens=10
# )

## Hard Prompt Tuning
config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="下面是一段人和机器的对话. 请根据人类输入的内容继续对话.",
    num_virtual_tokens=len(tokenizer("下面是一段人和机器的对话. 请根据人类输入的内容继续对话.")['input_ids']),
    tokenizer_name_or_path=model_id_or_path
)
```

这里需要涉及 soft 和 hard 的概念:
- Soft Prompt-Tuning
	- 只需要指定 `num_virtual_tokens` 个数
	- 此时 virtual_tokens 进行随机 initialize
	- loss 在初期下降很慢, 需要增加训练轮次
- Hard Prompt-Tuning
	- 需要指定如何对 virtual_tokens 进行 initialize
	- 此时 virtual_tokens 进行指定 initialize
	- loss 的下降较为明显
	- 此时的 `num_virtual_tokens` 通常会与 `prompt_tuning_init_text` tokenization 之后的长度保持一致
		- 如果 指定大小 *小于* `prompt_tuning_init_text` tokenization 之后的长度: 截断
		- 如果 指定大小 *大于* `prompt_tuning_init_text` tokenization 之后的长度: 循环填充

### 构建 PEFT Model

```python
from peft import get_peft_model
peft_model = get_peft_model(model, config)
peft_model
```

通过 `get_peft_model` 方法, 传入我们一开始加载的base模型, 会根据 config 信息构建 peft_model: 
```
PeftModelForCausalLM(
  (base_model): BloomForCausalLM(
    (transformer): BloomModel(
      (word_embeddings): Embedding(46145, 2048)
      (word_embeddings_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
      (h): ModuleList(
        (0-23): 24 x BloomBlock(
          (input_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          (self_attention): BloomAttention(
            (query_key_value): Linear(in_features=2048, out_features=6144, bias=True)
            (dense): Linear(in_features=2048, out_features=2048, bias=True)
            (attention_dropout): Dropout(p=0.0, inplace=False)
          )
          (post_attention_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
          (mlp): BloomMLP(
            (dense_h_to_4h): Linear(in_features=2048, out_features=8192, bias=True)
            (gelu_impl): BloomGelu()
            (dense_4h_to_h): Linear(in_features=8192, out_features=2048, bias=True)
          )
        )
      )
      (ln_f): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
    )
    (lm_head): Linear(in_features=2048, out_features=46145, bias=False)
  )
  (prompt_encoder): ModuleDict(
    (default): PromptEmbedding(
      (embedding): Embedding(16, 2048)
    )
  )
  (word_embeddings): Embedding(46145, 2048)
)
```

注意这里多为原本模型添加了 `PromptEmbedding` 的结构, 这也是我们整个模型会训练的部分. 查看一下训练模型参数量: 
```python
peft_model.print_trainable_parameters()
```

```
['trainable params: 32,768 || all params: 1,303,144,448 || trainable%: 0.0025\n']
```

## 模型训练

后续模型训练与训练模型的基本步骤保持一致, 需要注意的是此时是对 prompt_model 进行训练.

### 配置 TrainingArguments 和 Trainer

```python
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq

args = TrainingArguments(
    output_dir="../output/prompt_soft_tuning_bloom_1b4_zh",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=1,
)

trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
)
```

### 训练

```python
trainer.train()
```

```
TrainOutput(global_step=625, training_loss=2.353355191040039, metrics={'train_runtime': 923.1481, 'train_samples_per_second': 10.832, 'train_steps_per_second': 0.677, 'total_flos': 1.351054558347264e+16, 'train_loss': 2.353355191040039, 'epoch': 1.0})
```

### 推理
```python
instruction_text = "考试有哪些技巧?"
input_text = ""
ipt = tokenizer("Human: {}\n{}".format(instruction_text, input_text) + "\n\nAssistant: ", return_tensors="pt").to(peft_model.device)
print(tokenizer.decode(peft_model.generate(**ipt, max_length=100, do_sample=True)[0], skip_special_tokens=True))
```

```
['Human: 考试有哪些技巧?\n', '\n', '\n', 'Assistant: 在面对一道考试时，需要掌握哪些技巧？\n', '\n', '\n', '在面对数学考试时，建议考生以平常心来面对。数学考试是主观题，因此想要解好答案，必须熟记基本知识。但面对高难率的数学题，则需要使用适当的策略来思考题目，并快速将答案记下来。考生可以先看清题目中提到的概念，再思考出题者想要表达\n']
```

## 加载 Prompt-Tuning 后的模型

因为在进行 Prompt-Tuning 之后, ckpt 保存的是 `PromptEmbedding` 的部分, 我们需要借助 PeftModel 来进行模型加载: 

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# 加载原始模型
model_id_or_path = "../models/bloom-1b4-zh"
model = AutoModelForCausalLM.from_pretrained(model_id_or_path)

# 加载微调后的模型
ckpt_path = "../output/prompt_soft_tuning_bloom_1b4_zh/checkpoint-500"
peft_model = PeftModel.from_pretrained(model=model, model_id=ckpt_path)

```

推理: 
```python
instruction_text = "考试有哪些技巧?"
input_text = ""
ipt = tokenizer("Human: {}\n{}".format(instruction_text, input_text) + "\n\nAssistant: ", return_tensors="pt").to(peft_model.device)
print(tokenizer.decode(peft_model.generate(**ipt, max_length=100, do_sample=True)[0], skip_special_tokens=True))
```

```
['Human: 考试有哪些技巧?\n', '\n', '\n', 'Assistant: 考试技巧有很多：（1）阅读文章时，根据内容选择合适的单词，比如英语文章可以看句子，而汉语文章则相反，所以可以选用的单词不一定对，只有通过正确选择才能得高分。（2）阅读文章时，从题目到答案一步一步去看，不要跳跃，这样可以比较完整地了解选项以及整个文章逻辑情况。（3）阅读文章时，经常回顾所学的单词，这样可以\n']
```
