---
title: Huggingface-基础部分-03-DataSets 和 DataCollator
date: 2024-10-02 12:00:00
updated: 2024-10-02 12:00:00
cover: /img/huggingface.png
top_img: /img/huggingface_long.png
tags: Huggingface
categories: 算法
---

## 基本使用

1. 加载在线数据集(load_dataset)

2. 加载数据集某一项任务(load_dataset)

3. 按照数据集划分进行加载(load_dataset)

4. 查看数据集(index and slice)

5. 数据集划分(train_test_split)

6. 数据选取与过滤(select and filter)

7. 数据映射(map)

8. 保存与加载(save_to_disk / load_from_disk)

## 数据集加载

```python
# 加载在线数据集
datasets = load_dataset("madao33/new-title-chinese") -> DatasetDict

# 加载数据集合集的某一项任务
datasets = load_dataset("super_glue", "boolq") -> DatasetDict

# 按照数据集划分加载
datasets = load_dataset("madao33/new-title-chinese", split="train") -> Dataset
```
注意 `DatasetDict` 和 `Dataset` 两种数据类型, `DatasetDict` 存放了不同划分的 `Dataset`.

我们可以使用一些比较独特的切片方式, 传入 split 参数实现: 
```python
# split特殊切片
datasets = load_dataset("madao33/new-title-chinese", split="train[10:100]") -> Dataset
datasets = load_dataset("madao33/new-title-chinese", split="train[:10%]") -> Dataset
datasets = load_dataset("madao33/new-title-chinese", split=["train[:50%]", "validation[:10%]"]) -> List[Dataset]
```

两个数据集常用的查看方法, 同时在map方法中经常使用 `.column_names` 来删除处理前的列: 
```python
datasets["train"].column_names
datasets["train"].features
```

## 数据集划分

```python
test_datasets = datasets["train"].train_test_split(test_size=0.1)
```
此时数据由 `Dataset` 转变为划分好两个数据集的 `DatasetDict` 类型.

有关“分类任务数据集”: 
```python
# 分类任务数据集, 可以传入label列, 使划分均衡
# 但需要保证label列是ClassLabel类型

import copy
ClassLabel_datasets = copy.deepcopy(datasets)

from datasets import ClassLabel

class_labels = ClassLabel(num_classes=2, names=["negative", "positive"]) # 定义类别映射
ClassLabel_datasets["train"] = ClassLabel_datasets["train"].cast_column("label", class_labels) # 转换 label 列
  
print(ClassLabel_datasets["train"].features)
  
# tratify_by_column="label" 传入 label 列
ClassLabel_datasets = ClassLabel_datasets["train"].train_test_split(test_size=0.1, stratify_by_column="label")
```

## 数据选取与过滤

1. select方法 v.s. 索引
```python
# 选取部分数据集
datasets["train"].select([0, 1, 2])

# 相比直接索引可以保留数据集格式, datasets["train"][0, 1, 2], 返回的是字典
```

2. filter方法, 传入一个匿名函数
```python
# 过滤数据集
datasets["train"].filter(lambda example: example["review"] is not None and "携程" in example["review"])
```

## 数据映射: map方法, 主要使用
```python
# 一个简单例子
# 定义一个map函数, 给review列添加前缀: "评论: "

def add_prefix(example):
	if example["review"] is not None:
		example["review"] = "评论: " + example["review"]
	return example

prefix_datasets = test_datasets.map(add_prefix)
```

应用 tokenizer: 
```python
from transformers import AutoTokenizer

model_id_or_path = "/root/autodl-tmp/ianli/models/rbt3"
tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)

import torch

def tokenize_function(example):
	inputs = tokenizer(
		example["review"],
		truncation=True,
		max_length=512,
		padding="longest",
	)
	inputs["labels"] = example["label"]
	return inputs

# 过滤 None
test_datasets = test_datasets.filter(lambda example: example["review"] is not None)

# 应用 map函数
tokenized_datasets = test_datasets.map(tokenize_function, batched=True, remove_columns=test_datasets["train"].column_names)
```

注意map方法可以支持批处理, 传入 `batched=True` 但是此时的tokenizer需要是 FastTokenizer, 如果是 SlowTokenizer 传入这个参数是会报错的, 不过可以传入多线程的参数:
```python
tokenized_datasets = test_datasets.map(tokenize_function, num_proc=4)
```

## 保存与加载(序列化)

```python
tokenized_datasets.save_to_disk("/root/autodl-tmp/ianli/data/tokenized_datasets")

tokenized_datasets = load_from_disk("/root/autodl-tmp/ianli/data/tokenized_datasets")
```

## 直接加载本地数据集

假定去加载一个 csv文件: 
```python
dataset_id_or_path = "./ChnSentiCorp_htl_all.csv"
# 1: 加载后是一个 Dict
dataset = load_dataset("csv", data_files=dataset_id_or_path)

# 2: 加载后是一个 Set
dataset = load_dataset("csv", data_files=dataset_id_or_path, split="train")

# 3: 加载后是一个 Set
dataset = Dataset.from_csv(dataset_id_or_path)
```

加载文件夹内全部文件: 
```python
# 如果一个文件夹下都是csv文件, 可以直接指定文件夹路径, 注意参数是data_dir
dataset_id_or_path = "/root/autodl-tmp/ianli/data/test_csv"

dataset = load_dataset("csv", data_dir=dataset_id_or_path, split="train")
```

加载文件夹内部分文件: 
```python
data_list = [
	"/root/autodl-tmp/ianli/data/test_csv/ChnSentiCorp_htl_all_1.csv",
	"/root/autodl-tmp/ianli/data/test_csv/ChnSentiCorp_htl_all_2.csv",
	"/root/autodl-tmp/ianli/data/test_csv/ChnSentiCorp_htl_all_3.csv"
]
dataset = load_dataset("csv", data_files=data_list, split="train")
```

加载其他格式的数据, 例如 list, pandas:
```python
# Dataset.from_list 还支持 .from_pandas 等方法

# 需要注意的是, 如果数据已开始是一个list, 例如
data = ["abc", "def"]

# 则需要转化为
data = [{"text": "abc"}, {"text": "def"}]

# 才可以使用
Dataset.from_list(data)
```

## 自定义加载脚本: 边处理复杂数据边加载
```python
# 定义一个数据加载脚本
import datasets
from datasets import DownloadManager, DatasetInfo

# 定义一个处理类, 继承 datasets.GeneratorBaseBuilder 并完善下述三个方法
class MyData(datasets.GeneratorBaseBuilder):
	
	def _info(self):
		pass
	
	def _split_generators(self, dl_manager: DownloadManager):
		pass
	  
	def _generate_examples(self, filepath):
		pass

# 使用 数据加载脚本 加载数据集
dataset = load_dataset("my_data.py", split="train")
```

示例: 
```python
import json
import datasets
from datasets import DownloadManager, DatasetInfo


class CMRC2018TRIAL(datasets.GeneratorBasedBuilder):

    def _info(self) -> DatasetInfo:
        """
            info方法, 定义数据集的信息,这里要对数据的字段进行定义
        :return:
        """
        return datasets.DatasetInfo(
            description="CMRC2018 trial",
            features=datasets.Features({
                    "id": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    )
                })
        )

    def _split_generators(self, dl_manager: DownloadManager):
        """
            返回datasets.SplitGenerator
            涉及两个参数: name和gen_kwargs
            name: 指定数据集的划分
            gen_kwargs: 指定要读取的文件的路径, 与_generate_examples的入参数一致
        :param dl_manager:
        :return: [ datasets.SplitGenerator ]
        """
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, 
                                        gen_kwargs={"filepath": "./cmrc2018_trial.json"})]

    def _generate_examples(self, filepath):
        """
            生成具体的样本, 使用yield
            需要额外指定key, id从0开始自增就可以
        :param filepath:
        :return:
        """
        # Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for example in data["data"]:
                for paragraph in example["paragraphs"]:
                    context = paragraph["context"].strip()
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = qa["id"]

                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"].strip() for answer in qa["answers"]]

                        yield id_, {
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
```

## DataCollator

> 在进行 DataLoader 的过程中, 构成 batch 的过程中可能会存在对数据的一些处理, 例如 根据当前 batch 中样本长度动态 padding. 
> 
> 这个时候我们通常需要在 DataLoader 中传入一个 collator_fn 来实现.
> 
> 如果数据中有除了: \['input_ids', 'token_type_ids', 'attention_mask', 'labels'\] 以外的自定义的字段, 官方自带的 Collator 是无法自动进行padding的, 需要自己实现 collator_fn.

首先预处理 DataSet: 
```python
from transformers import AutoTokenizer

model_id_or_path = "/root/autodl-tmp/ianli/models/rbt3"
tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)

# 定义一个map函数, tokenization
def tokenize_function(example):
	inputs = tokenizer(
		example["review"],
		truncation=True,
		max_length=512,
		# 此时先不进行padding, 之后会使用DataCollatorWithPadding进行padding
		# 也就是组成batch时再进行padding, 根据batch动态padding
		# padding="longest",
	)
	inputs["labels"] = example["label"]
	return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
```

定义 collator_fn, transformers中有一些自定义的:
```python
# 实例化一个DataCollatorWithPadding对象
collator = DataCollatorWithPadding(tokenizer=tokenizer)

from torch.utils.data import DataLoader
dl = DataLoader(tokenized_datasets, batch_size=4, collate_fn=collator, shuffle=True)
```

我们可以查看一下每一批次的 batch 的 tokens 长度是动态变化的:
```python
num = 0
for batch in dl:
	print(batch["input_ids"].size())
	num += 1
	if num > 10:
		break
```

```
torch.Size([4, 237]) 
torch.Size([4, 80]) 
torch.Size([4, 87]) 
torch.Size([4, 512]) 
torch.Size([4, 173]) 
torch.Size([4, 181]) 
torch.Size([4, 251]) 
torch.Size([4, 207]) 
torch.Size([4, 103]) 
torch.Size([4, 198]) 
torch.Size([4, 177])
```