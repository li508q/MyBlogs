---
title: 分布式-02-DeepSpeed
date: 2024-10-30 12:00:00
tags: 分布式训练
categories: 系统
---

> **Accelerate** 集成了 **DeepSpeed ZeRO** 的所有功能。这包括 **ZeRO** 的第 1、2 和 3 阶段，以及 **ZeRO-Offload**、**ZeRO-Infinity**（可以卸载到 Disk/NVMe）和 **ZeRO++**。
> 
> Huggingface中的 **Accelerate** 通过两种选项集成了 DeepSpeed：
>	1. **通过 deepspeed 配置文件**
>	2. **通过 deepspeed_plugin**

---

## 如何通过Accelerate使用DeepSpeed

### 1. 通过 deepspeed_plugin

在您的机器上运行：

```bash
accelerate config
```

并回答所提出的问题。它会询问您是否要为 **DeepSpeed** 使用配置文件，您应该回答否。然后回答以下问题以生成一个基本的 **DeepSpeed** 配置。这将生成一个配置文件，在执行以下命令时将自动使用该配置文件来正确设置默认选项：

```bash
accelerate launch my_script.py --args_to_my_script
```

例如，这里是如何使用 **DeepSpeed Plugin** 运行 NLP 示例 `examples/nlp_example.py`（从仓库根目录）：

#### ZeRO Stage-2 DeepSpeed Plugin 示例

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: true
  zero_stage: 2
distributed_type: DEEPSPEED
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
```

```bash
accelerate launch examples/nlp_example.py --mixed_precision fp16
```

#### ZeRO Stage-3 使用 CPU Offload 的 DeepSpeed Plugin 示例

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
```

```bash
accelerate launch examples/nlp_example.py --mixed_precision fp16
```

目前，**Accelerate** 通过 CLI 支持以下配置：

- `zero_stage`: [0] 禁用， [1] 优化器状态分区， [2] 优化器+梯度状态分区， [3] 优化器+梯度+参数分区
- `gradient_accumulation_steps`: 在平均和应用梯度之前要累积的训练步骤数。
- `gradient_clipping`: 启用梯度裁剪并设定值。
- `offload_optimizer_device`: [none] 禁用优化器卸载， [cpu] 将优化器卸载到 CPU， [nvme] 将优化器卸载到 NVMe SSD。仅适用于 ZeRO >= Stage-2。
- `offload_optimizer_nvme_path`: 决定将优化器状态卸载到的 NVMe 路径。如果未指定，则默认为 'none'。
- `offload_param_device`: [none] 禁用参数卸载， [cpu] 将参数卸载到 CPU， [nvme] 将参数卸载到 NVMe SSD。仅适用于 ZeRO Stage-3。
- `offload_param_nvme_path`: 决定将参数卸载到的 NVMe 路径。如果未指定，则默认为 'none'。
- `zero3_init_flag`: 决定是否启用 `deepspeed.zero.Init` 来构建大规模模型。仅适用于 ZeRO Stage-3。
- `zero3_save_16bit_model`: 决定在使用 ZeRO Stage-3 时是否保存 16 位模型权重。
- `mixed_precision`: `no` 表示 FP32 训练， `fp16` 表示 FP16 混合精度训练， `bf16` 表示 BF16 混合精度训练。
- `deepspeed_moe_layer_cls_names`: 逗号分隔的 transformer Mixture-of-Experts (MoE) 层类名列表（区分大小写），例如 `MixtralSparseMoeBlock`, `Qwen2MoeSparseMoeBlock`, `JetMoEAttention,JetMoEBlock` ...
- `deepspeed_hostfile`: 用于配置多节点计算资源的 **DeepSpeed** hostfile。
- `deepspeed_exclusion_filter`: 使用多节点设置时的 **DeepSpeed** 排除过滤字符串。
- `deepspeed_inclusion_filter`: 使用多节点设置时的 **DeepSpeed** 包含过滤字符串。
- `deepspeed_multinode_launcher`: 要使用的 **DeepSpeed** 多节点启动器。如果未指定，则默认为 `pdsh`。
- `deepspeed_config_file`: **DeepSpeed** 配置文件的路径，格式为 `json`。有关更多详细信息，请参见下一节。

要调整更多选项，您需要使用 **DeepSpeed** 配置文件。

### 2. DeepSpeed 配置文件

在您的机器上运行：

```bash
accelerate config
```

并回答所提出的问题。它会询问您是否要为 **DeepSpeed** 使用配置文件，您应该回答是并提供 **DeepSpeed 配置文件** 的路径。这将生成一个配置文件，在执行以下命令时将自动使用该配置文件来正确设置默认选项：

```bash
accelerate launch my_script.py --args_to_my_script
```

例如，这里是如何使用 **DeepSpeed 配置文件** 运行 NLP 示例 `examples/by_feature/deepspeed_with_config_support.py`（从仓库根目录）：

#### ZeRO Stage-2 DeepSpeed 配置文件示例

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: /home/ubuntu/accelerate/examples/configs/deepspeed_config_templates/zero_stage2_config.json
  zero3_init_flag: true
distributed_type: DEEPSPEED
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
```

**zero_stage2_config.json** 的内容如下：

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "weight_decay": "auto",
            "torch_adam": true,
            "adam_w_mode": true
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": "auto",
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": 1,
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

运行命令：

```bash
accelerate launch examples/by_feature/deepspeed_with_config_support.py \
--config_name "gpt2-large" \
--tokenizer_name "gpt2-large" \
--dataset_name "wikitext" \
--dataset_config_name "wikitext-2-raw-v1" \
--block_size 128 \
--output_dir "./clm/clm_deepspeed_stage2_accelerate" \
--learning_rate 5e-4 \
--per_device_train_batch_size 24 \
--per_device_eval_batch_size 24 \
--num_train_epochs 3 \
--with_tracking \
--report_to "wandb"
```

#### ZeRO Stage-3 使用 CPU Offload 的 DeepSpeed 配置文件示例

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: /home/ubuntu/accelerate/examples/configs/deepspeed_config_templates/zero_stage3_offload_config.json
  zero3_init_flag: true
distributed_type: DEEPSPEED
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
```

**zero_stage3_offload_config.json** 的内容如下：

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "weight_decay": "auto"
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "sub_group_size": 1e9,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": "auto"
    },
    "gradient_accumulation_steps": 1,
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

运行命令：

```bash
accelerate launch examples/by_feature/deepspeed_with_config_support.py \
--config_name "gpt2-large" \
--tokenizer_name "gpt2-large" \
--dataset_name "wikitext" \
--dataset_config_name "wikitext-2-raw-v1" \
--block_size 128 \
--output_dir "./clm/clm_deepspeed_stage3_offload_accelerate" \
--learning_rate 5e-4 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--num_train_epochs 3 \
--with_tracking \
--report_to "wandb"
```

---

**说明：**

- **ZeRO Stage-2 配置文件** 示例展示了如何配置 **DeepSpeed** 以使用 ZeRO 的第 2 阶段优化，包括梯度累积步数、梯度裁剪、优化器类型等参数。

- **ZeRO Stage-3 使用 CPU Offload 的配置文件** 示例则进一步展示了如何将优化器状态和模型参数卸载到 CPU，以支持更大规模模型的训练。

- 运行命令中使用的 `--config_name`、`--tokenizer_name`、`--dataset_name` 等参数保持不变，确保 **Accelerate** 能正确加载和使用指定的配置文件。

- 通过配置文件方式，用户可以更灵活地调整 **DeepSpeed** 的各种参数，以适应不同的训练需求和硬件环境。

---

## **ZeRO++ 配置示例**

您可以通过使用适当的配置参数来使用 **ZeRO++** 的功能。请注意，**ZeRO++** 是 **ZeRO Stage 3** 的扩展。以下是如何修改配置文件的示例，摘自 **DeepSpeed** 的 **ZeRO++** 教程：

```json
{
    "zero_optimization": {
        "stage": 3,
        "reduce_bucket_size": "auto",

        "zero_quantized_weights": true,
        "zero_hpz_partition_size": 8,
        "zero_quantized_gradients": true,

        "contiguous_gradients": true,
        "overlap_comm": true
    }
}
```

对于分层分区，`zero_hpz_partition_size` 的分区大小理想情况下应设置为每个节点的 GPU 数量。（例如，上述配置文件假设每个节点有 8 个 GPU）

### **使用 DeepSpeed 配置文件时的重要代码更改**

**DeepSpeed Optimizers** 和 **Schedulers**。有关更多信息，请参阅 [DeepSpeed Optimizers](https://www.deepspeed.ai/docs/config-json/#optimizer) 和 [DeepSpeed Schedulers](https://www.deepspeed.ai/docs/config-json/#scheduler) 文档。我们将查看在使用这些时代码需要进行的更改。

#### a. **DS Optim + DS Scheduler**

当 **DeepSpeed** 配置文件中同时存在 `optimizer` 和 `scheduler` 键时。在这种情况下，将使用配置文件中的优化器和调度器，用户需要使用 `accelerate.utils.DummyOptim` 和 `accelerate.utils.DummyScheduler` 来替换代码中的 PyTorch/自定义优化器和调度器。以下是来自 `examples/by_feature/deepspeed_with_config_support.py` 的代码片段：

```python
# 如果配置文件中指定了 `optimizer`，则创建 Dummy Optimizer，否则创建 Adam Optimizer
optimizer_cls = (
    torch.optim.AdamW
    if accelerator.state.deepspeed_plugin is None
    or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
    else DummyOptim
)
optimizer = optimizer_cls(optimizer_grouped_parameters, lr=args.learning_rate)

# 如果配置文件中未指定 `scheduler`，则创建 `args.lr_scheduler_type` Scheduler，否则创建 Dummy Scheduler
if (
    accelerator.state.deepspeed_plugin is None
    or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
):
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
else:
    lr_scheduler = DummyScheduler(
        optimizer, total_num_steps=args.max_train_steps, warmup_num_steps=args.num_warmup_steps
    )
```

#### b. **Custom Optim + Custom Scheduler**

当 **DeepSpeed** 配置文件中同时缺少 `optimizer` 和 `scheduler` 键时。在这种情况下，用户无需更改任何代码，这是使用 **DeepSpeed Plugin** 集成时的情况。在上述示例中，如果配置文件中缺少 `optimizer` 和 `scheduler` 键，代码将保持不变。

#### c. **Custom Optim + DS Scheduler**

当 **DeepSpeed** 配置文件中仅存在 `scheduler` 键时。在这种情况下，用户必须使用 `accelerate.utils.DummyScheduler` 来替换代码中的 PyTorch/自定义调度器。

#### d. **DS Optim + Custom Scheduler**

当 **DeepSpeed** 配置文件中仅存在 `optimizer` 键时。这将导致错误，因为只能在使用 **DS Optim** 时使用 **DS Scheduler**。

### **注意事项**

请注意上述示例 **DeepSpeed** 配置文件中的 `auto` 值。这些值由 `prepare` 方法根据模型、数据加载器、Dummy Optimizer 和 Dummy Schedulers 自动处理。仅示例中指定的 `auto` 字段由 `prepare` 方法处理，其余字段必须由用户显式指定。

**自动值的计算方式如下：**

- `reduce_bucket_size`: `hidden_size * hidden_size`
- `stage3_prefetch_bucket_size`: `int(0.9 * hidden_size * hidden_size)`
- `stage3_param_persistence_threshold`: `10 * hidden_size`

为了使这 3 个配置项的自动功能正常工作，**Accelerate** 将使用 `model.config.hidden_size` 或 `max(model.config.hidden_sizes)` 作为 `hidden_size`。如果这两个都不可用，启动将失败，您需要手动设置这 3 个配置项。请记住，前两个配置项是通信缓冲区——它们越大，通信效率越高，但也会消耗更多的 GPU 内存，因此这是一个可调节的性能权衡。

---

### **使用 DeepSpeed 配置文件时需要注意的事项**

以下是一个在不同场景下使用 `deepspeed_config_file` 的示例脚本。

**代码 `test.py`：**

```python
from accelerate import Accelerator
from accelerate.state import AcceleratorState


def main():
    accelerator = Accelerator()
    accelerator.print(f"{AcceleratorState()}")


if __name__ == "__main__":
    main()
```

#### **场景 1：手动修改的 accelerate 配置文件，同时包含 `deepspeed_config_file` 及其他条目**

**accelerate 配置内容：**

```yaml
command_file: null
commands: null
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: 'cpu'
  offload_param_device: 'cpu'
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
  deepspeed_config_file: 'ds_config.json'
distributed_type: DEEPSPEED
downcast_bf16: 'no'
dynamo_backend: 'NO'
fsdp_config: {}
gpu_ids: null
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
megatron_lm_config: {}
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_name: null
tpu_zone: null
use_cpu: false
```

**`ds_config.json` 内容：**

```json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "stage3_gather_16bit_weights_on_model_save": false,
        "offload_optimizer": {
            "device": "none"
        },
        "offload_param": {
            "device": "none"
        }
    },
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": 10,
    "steps_per_print": 2000000
}
```

**运行命令 `accelerate launch test.py` 的输出：**

```
ValueError: When using `deepspeed_config_file`, the following accelerate config variables will be ignored:
['gradient_accumulation_steps', 'gradient_clipping', 'zero_stage', 'offload_optimizer_device', 'offload_param_device',
'zero3_save_16bit_model', 'mixed_precision'].
Please specify them appropriately in the DeepSpeed config file.
If you are using an accelerate config file, remove other config variables mentioned in the above specified list.
The easiest method is to create a new config following the questionnaire via `accelerate config`.
It will only ask for the necessary config variables when using `deepspeed_config_file`.
```

#### **场景 2：使用错误解决方案创建新的 accelerate 配置，并检查不再抛出模糊错误**

**运行 `accelerate config`：**

```
$ accelerate config
-------------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine
-------------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 
Do you wish to optimize your script with torch dynamo?[yes/NO]: 
Do you want to use DeepSpeed? [yes/NO]: yes
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: yes
Please enter the path to the json DeepSpeed config file: ds_config.json
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: yes
How many GPU(s) should be used for distributed training? [1]:4
accelerate configuration saved at ds_config_sample.yaml
```

**新的 accelerate 配置内容：**

```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: ds_config.json
  zero3_init_flag: true
distributed_type: DEEPSPEED
downcast_bf16: 'no'
dynamo_backend: 'NO'
fsdp_config: {}
machine_rank: 0
main_training_function: main
megatron_lm_config: {}
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
use_cpu: false
```

**运行命令 `accelerate launch test.py` 的输出：**

```
Distributed environment: DEEPSPEED  Backend: nccl
Num processes: 4
Process index: 0
Local process index: 0
Device: cuda:0
Mixed precision type: bf16
ds_config: {'bf16': {'enabled': True}, 'zero_optimization': {'stage': 3, 'stage3_gather_16bit_weights_on_model_save': False, 'offload_optimizer': {'device': 'none'}, 'offload_param': {'device': 'none'}}, 'gradient_clipping': 1.0, 'train_batch_size': 'auto', 'train_micro_batch_size_per_gpu': 'auto', 'gradient_accumulation_steps': 10, 'steps_per_print': inf, 'fp16': {'enabled': False}}
```

#### **场景 3：在 DeepSpeed 配置文件中将与 DeepSpeed 命令相关的 accelerate launch 命令参数设置为 "auto"，并检查是否按预期工作**

**新的 `ds_config.json`，将与 accelerate launch DeepSpeed 命令参数设置为 "auto"：**

```json
{
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": "auto",
        "stage3_gather_16bit_weights_on_model_save": "auto",
        "offload_optimizer": {
            "device": "auto"
        },
        "offload_param": {
            "device": "auto"
        }
    },
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "steps_per_print": 2000000
}
```

**运行命令：**

```bash
accelerate launch --mixed_precision="fp16" --zero_stage=3 --gradient_accumulation_steps=5 --gradient_clipping=1.0 --offload_param_device="cpu" --offload_optimizer_device="nvme" --zero3_save_16bit_model="true" test.py
```

**输出：**

```
Distributed environment: DEEPSPEED  Backend: nccl
Num processes: 4
Process index: 0
Local process index: 0
Device: cuda:0
Mixed precision type: fp16
ds_config: {'bf16': {'enabled': False}, 'zero_optimization': {'stage': 3, 'stage3_gather_16bit_weights_on_model_save': True, 'offload_optimizer': {'device': 'nvme'}, 'offload_param': {'device': 'cpu'}}, 'gradient_clipping': 1.0, 'train_batch_size': 'auto', 'train_micro_batch_size_per_gpu': 'auto', 'gradient_accumulation_steps': 5, 'steps_per_print': inf, 'fp16': {'enabled': True, 'auto_cast': True}}
```

**注意：**

- 剩余的 "auto" 值在调用 `accelerator.prepare()` 时处理，如使用 DeepSpeed 配置文件时重要代码更改中的第 2 点所述。
- 仅当 `gradient_accumulation_steps` 为 auto 时，创建 `Accelerator` 对象时传递的值（如 `Accelerator(gradient_accumulation_steps=k)`）将被使用。
- 使用 DeepSpeed Plugin 时，将使用 DeepSpeed Plugin 中的值，并覆盖在创建 `Accelerator` 对象时传递的值。

#### **保存和加载**

- **ZeRO Stage-1 和 Stage-2** 的模型保存和加载方式保持不变。
  
- **ZeRO Stage-3** 下，`state_dict` 仅包含占位符，因为模型权重被分区到多个 GPU 上。**ZeRO Stage-3** 有两种选项：

  a. **保存整个 16 位模型权重**，以便稍后使用 `model.load_state_dict(torch.load(pytorch_model.bin))` 直接加载。为此，可以在 DeepSpeed 配置文件中设置 `zero_optimization.stage3_gather_16bit_weights_on_model_save` 为 `True`，或在 DeepSpeed Plugin 中设置 `zero3_save_16bit_model` 为 `True`。请注意，此选项需要在一个 GPU 上合并权重，可能会很慢且占用大量内存，因此仅在需要时使用。以下是来自 `examples/by_feature/deepspeed_with_config_support.py` 的代码片段：

    ```python
    unwrapped_model = accelerator.unwrap_model(model)
    
    # 新代码 #
    # 如果 DeepSpeed 配置文件中的 `stage3_gather_16bit_weights_on_model_save` 为 True 或
    # DeepSpeed Plugin 中的 `zero3_save_16bit_model` 为 True，则在 ZeRO Stage-3 中将整个/未分区的 fp16 模型保存到输出目录。
    # 对于 ZeRO Stage 1 和 2，模型按通常方式保存在输出目录中。
    # 保存的模型名称为 `pytorch_model.bin`
    unwrapped_model.save_pretrained(
        args.output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )
    ```

  b. **获取 32 位权重**，首先使用 `model.save_checkpoint()` 保存模型。以下是来自 `examples/by_feature/deepspeed_with_config_support.py` 的代码片段：

    ```python
    success = model.save_checkpoint(PATH, ckpt_id, checkpoint_state_dict)
    status_msg = f"checkpointing: PATH={PATH}, ckpt_id={ckpt_id}"
    if success:
        logging.info(f"Success {status_msg}")
    else:
        logging.warning(f"Failure {status_msg}")
    ```

    这将在检查点目录中创建 ZeRO 模型和优化器分区以及 `zero_to_fp32.py` 脚本。您可以使用此脚本进行离线合并，无需配置文件或 GPU。以下是其使用示例：

    ```bash
    $ cd /path/to/checkpoint_dir
    $ ./zero_to_fp32.py . pytorch_model.bin
    Processing zero checkpoint at global_step1
    Detected checkpoint of type zero stage 3, world_size: 2
    Saving fp32 state dict to pytorch_model.bin (total_numel=60506624)
    ```

    **获取用于保存/推理的 32 位模型，您可以执行以下操作：**

    ```python
    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
    
    unwrapped_model = accelerator.unwrap_model(model)
    fp32_model = load_state_dict_from_zero_checkpoint(unwrapped_model, checkpoint_dir)
    ```

    **如果您只对 `state_dict` 感兴趣，可以执行以下操作：**

    ```python
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    
    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
    ```

    请注意，所有这些函数需要约 2 倍于最终检查点大小的内存（通用 RAM）。

#### **ZeRO 推理**

**DeepSpeed ZeRO Inference** 支持 ZeRO Stage 3 与 **ZeRO-Infinity**。它使用与训练相同的 ZeRO 协议，但不使用优化器和学习率调度器，仅 Stage 3 相关。通过 **Accelerate** 集成，您只需按如下所示准备模型和数据加载器：

```python
model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
```

#### **需要注意的几个问题**

- 当前的集成不支持 **DeepSpeed** 的 **Pipeline Parallelism**。
- 当前的集成不支持 **mpu**，限制了在 **Megatron-LM** 中支持的张量并行。
- 当前的集成不支持多个模型。

#### **DeepSpeed 资源**

与 DeepSpeed 相关的内部文档可以在这里找到：

- [项目的 GitHub](https://github.com/microsoft/DeepSpeed)
- [使用文档](https://www.deepspeed.ai/docs/)
- [API 文档](https://www.deepspeed.ai/docs/api/)
- [博客文章](https://www.deepspeed.ai/blog/)
- **论文：**
  - **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**
  - **ZeRO-Offload: Democratizing Billion-Scale Model Training**
  - **ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning**
  - **ZeRO++: Extremely Efficient Collective Communication for Giant Model Training**

最后，请记住，**Accelerate** 仅集成了 **DeepSpeed**，因此如果您在使用 **DeepSpeed** 时遇到任何问题或有任何疑问，请在 **DeepSpeed GitHub** 上提交问题。
