---
title: Huggingface-基础部分-02-Tokenizer
date: 2024-11-13 12:00:00
updated: 2024-11-13 12:00:00
cover: /img/huggingface.png
top_img: /img/huggingface_long.png
tags: Huggingface
categories: 算法
---

## 基本使用

在自然语言处理(NLP)任务中, 文本需要被模型理解和处理, 而最关键的一步就是“分词” (tokenization). HuggingFace 提供了非常强大且易用的分词工具, 能够快速实现从文本到模型可处理的张量(tensor)的转换: 

1. 加载保存(from_pretrained / save_pretrained)

2. 句子分词(tokenize)

3. 查看词典(vocab)

4. 索引转换(convert_tokens_to_ids / convert_ids_to_tokens)

5. 填充截断(padding / truncation)

6. 其他输入(attention_mask / token_type_ids)

```python
from transformers import AutoTokenizer
sen = "弱小的我也有大梦想!"

# Step1 加载与保存
# 从HuggingFace加载，输入模型名称，即可加载对于的分词器
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")

# tokenizer 保存到本地
tokenizer.save_pretrained("./roberta_tokenizer")

# 从本地加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("./roberta_tokenizer/")

# Step2 句子分词

tokens = tokenizer.tokenize(sen)

# Step3 查看词典
tokenizer.vocab
tokenizer.vocab_size

# Step4 索引转换
# 将词序列转换为id序列
ids = tokenizer.convert_tokens_to_ids(tokens)

# 将id序列转换为token序列
tokens = tokenizer.convert_ids_to_tokens(ids)

# 将token序列转换为string
str_sen = tokenizer.convert_tokens_to_string(tokens)

# 更便捷的实现方式
# 将字符串转换为id序列，又称之为编码
ids = tokenizer.encode(sen, add_special_tokens=True)

# 将id序列转换为字符串，又称之为解码
str_sen = tokenizer.decode(ids, skip_special_tokens=False)

# Step5 填充与截断
# 填充
ids = tokenizer.encode(sen, padding="max_length", max_length=15)

# 截断
ids = tokenizer.encode(sen, max_length=5, truncation=True)

# Step6 其他输入部分: attention_mask, token_type_ids
ids = tokenizer.encode(sen, padding="max_length", max_length=15)
attention_mask = [1 if idx != 0 else 0 for idx in ids]
token_type_ids = [0] * len(ids)

# Step7 调用方法实现Step6: encode_plus方法
inputs = tokenizer.encode_plus(sen, padding="max_length", max_length=15)
# 默认tokenizer调用的就是encode_plus, 这也是我们最终使用的部分
inputs = tokenizer(sen, padding="max_length", max_length=15)

# Step8 处理batch数据, tokenizer可以直接处理list形式的batch
sens = ["弱小的我也有大梦想",
        "有梦想谁都了不起",
        "追逐梦想的心，比梦想本身，更可贵"]
res = tokenizer(sens)
```

我们可以测试一下: 

```python
%%time
# 单条循环处理
for i in range(1000):
    tokenizer(sen)
```

```python
%%time
# 处理batch数据
res = tokenizer([sen] * 1000)
```

## Fast / Slow Tokenizer

有一个值得大家关注的地方是:
- FastTokenizer(多数模型默认此方法)
	- 基于 Rust 实现, 速度快
	- 额外的方法: offsets_mapping, word_ids
- SlowTokenizer
	- 基于 Python 实现, 速度慢

```python
sen = "弱小的我也有大Dreaming!"

fast_tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")

slow_tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese", use_fast=False)
```

可以通过以下来测试一下处理速度:

```python
%%time
# 单条循环处理
for i in range(10000):
    fast_tokenizer(sen)
```

```python
%%time
# 单条循环处理
for i in range(10000):
    slow_tokenizer(sen)
```

```python
%%time
# 处理batch数据
res = fast_tokenizer([sen] * 10000)
```

```python
%%time
# 处理batch数据
res = slow_tokenizer([sen] * 10000)
```

FastTokenizer提供了一些额外的方法, 例如指定 `return_offsets_mapping=True`
```python
inputs = fast_tokenizer(sen, return_offsets_mapping=True)
```

此时 inputs 会存在两个keys, return_offsets_mapping 和 word_ids
```python
inputs['return_offsets_mapping']
inputs['word_ids']  # 可以分辨出来可能存在一个词被分成了两个或以上的tokens
```
- offsets_mapping: 在对文本进行分词后, `offset_mapping` 可以告诉我们每个 token 在原始字符串中对应的 **起始位置**和**结束位置**(即字符级别的位置信息): 
	- 例如: `offset_mapping = [(0, 1), (1, 2), (3, 10)...]`.
	- 这在进行高亮显示、切片还原原文本等需要精确字符位置的操作时尤其有用.
- word_ids: 该方法会返回一个列表, 指示每个 token 所属的 **单词下标**: 
	- 例如, 文本 `"Hello World"` 被分为三段 token `[Hello, Wor, ##ld]`, 对应的 word_ids 可能是 `[0, 1, 1]` (0 表示第一个单词，1 表示第二个单词).
	- 在需要把多个子词拼回成同一个原词(或者区分不同单词)时非常方便.

## 特殊Tokenizer的加载

有一些模型的 tokenizer 是以一个实现的 py 文件存放的
这时需要传入 `trust_remote_code=True` 才能够正常加载 tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-13B-base", trust_remote_code=True)

tokenizer.save_pretrained("skywork_tokenizer")
```