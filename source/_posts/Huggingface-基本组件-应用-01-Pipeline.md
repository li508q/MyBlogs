---
title: Huggingface-基本组件-应用-01-Pipeline
date: 2024-11-10 12:00:00
updated: 2024-11-10 12:00:00
cover: /img/huggingface.png
top_img: /img/huggingface_long.png
tags: Huggingface
categories: 框架
---

## Pipeline背后实现

  > Pipeline 的主要作用是简化了一些常用推理的实现, 其实现的功能, 背后的逻辑如下: 

1. 初始化 Tokenizer
	- tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)

2. 初始化 Model
	- model = AutoModelForSequenceClassification.from_pretrained(model_id_or_path)

3. 数据预处理
	- input_text = "I love you"
	- input_ids = tokenizer.encode(input_text, return_tensors='pt')

4. 模型预测
	- res_logits = model(input_ids).logits

5. 结果后处理
	- pred = torch.argmax(torch.softmax(res_logits, dim=-1)).item()
	- result = model.config.id2label\[pred\]

## 使用示例

我们可以通过: 

```python
from transformers.pipelines import SUPPORTED_TASKS

for k, v in SUPPORTED_TASKS.items():
	print(k, v)
```

来查看 Huggingface 中的 Pipeline 支持哪些任务.

如果我们使用其实现一个“机器阅读理解”任务, 在 Hugging Face Transformers 中，`pipeline("question-answering")` 是一个用于「机器阅读理解（MRC）」任务的高阶接口。使用这个 Pipeline 时，你需要提供一个 **question**（问题）和一个 **context**（上下文）作为输入，模型会在上下文中寻找并输出最可能回答该问题的文本片段:

```python
from transformers import pipeline
model_id_or_path = "../models/Qwen2-1.5B-Instruct"

pipe = pipeline("question-answering", model=model_id_or_path, tokenizer=model_id_or_path, device=0)

pipe(question="中国的首都是哪里？", context="中国的首都是北京。", max_answer_len=1)
```

**输入参数说明**

- **question**：字符串，表示需要回答的问题。
- **context**：字符串，模型将从中抽取答案的文本。
- **max_answer_len**：用于限制答案的最大长度（以 token 计）；如果答案过长，会在此参数所设的长度处截断。
- **device**：指定使用 CPU（device=-1）还是 GPU（device=0 表示使用第一块 GPU）。
- （其他如 `top_k`, `handle_impossible_answer` 等可选参数，也可以影响输出的答案个数、格式，以及模型处理不可回答问题的方式。）

**结果解释**  

得到的输出是：

```
{'score': 0.7321616530418396, 'start': 6, 'end': 8, 'answer': '北'}
```

这个返回结果是一个包含以下键值对的字典：

1. **answer**：模型抽取的答案文本。这里是「中国的首都是北京」。
2. **score**：模型对该答案的置信度分数，数值越高表示模型越确定这是正确答案。
3. **start** 和 **end**：在 `context` 字符串里，答案对应起始和结束的字符索引。

## 其他Pipeline示例

实际上 Huggingface 中提供的 Pipeline 不止能够支持 NLP 相关的任务, CV 任务也可以使用, 例如 `zero-shot-object-detection` 就可以实现对图片的目标检测. 这部分的内容后续有时间, 会添加一个示例. 