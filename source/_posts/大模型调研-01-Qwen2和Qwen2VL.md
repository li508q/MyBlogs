---
title: 大模型调研-Qwen2和Qwen2VL
top: 999
date: 2024-12-15 12:00:00
updated: 2024-12-15 12:00:00
cover: /img/huggingface.png
top_img: /img/huggingface_long.png
tags: 大模型调研
categories: 算法
---

# 模型结构

## `Qwen2-1.5B` 模型结构

```scss
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(151936, 1536)
    (layers): ModuleList(
      (0-27): 28 x Qwen2DecoderLayer(
        (self_attn): Qwen2SdpaAttention(
          (q_proj): Linear(in_features=1536, out_features=1536, bias=True)
          (k_proj): Linear(in_features=1536, out_features=256, bias=True)
          (v_proj): Linear(in_features=1536, out_features=256, bias=True)
          (o_proj): Linear(in_features=1536, out_features=1536, bias=False)
          (rotary_emb): Qwen2RotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=1536, out_features=8960, bias=False)
          (up_proj): Linear(in_features=1536, out_features=8960, bias=False)
          (down_proj): Linear(in_features=8960, out_features=1536, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((1536,), eps=1e-06)
    (rotary_emb): Qwen2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=1536, out_features=151936, bias=False)
)
```

`Qwen2ForCausalLM` 模型主要由两大核心组件构成：

1. **模型（`model`）**：基于 Transformer 的核心架构，负责处理输入 Token 并生成上下文嵌入。
2. **语言建模头（`lm_head`）**：将模型输出的嵌入转换为对应词汇的 Logits，支持 Token 预测。

### 值得注意的地方

1. `Qwen2RotaryEmbedding`: 在注意力机制中使用**旋转位置编码**
2. `Qwen2MLP`: 在MLP中使用了 **门控 MLP（Gated MLP）** 架构 
	- `up_proj` 主要负责扩展特征空间，使模型能够学习更复杂的表示
	- `gate_proj` 主要生成控制信息流的门控信号
	- 升维过程中的
	- 生成门控信号，用于调节 MLP 内部的信息流
3. `Qwen2RMSNorm`: 归一化使用 **RMSNorm（均方根归一化）** 
	- RMSNorm 仅基于均方根（Root Mean Square, RMS）来进行归一化，而不计算均值。
	- 对于输入向量 $\mathbf{x}$，RMSNorm 的计算公式为：
	$$
	RMSNorm(x)=γ(xRMS(x)+ϵ)+β\text{RMSNorm}(\mathbf{x}) = \gamma \left( \frac{\mathbf{x}}{\text{RMS}(\mathbf{x}) + \epsilon} \right) + \beta
	$$
		其中，$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2}$，$\gamma$ 和 $\beta$ 同样是可学习参数，$\epsilon$ 防止分母为零。
	- **特点**：
		- **计算效率**：RMSNorm 省去了均值的计算，降低了计算复杂度，特别是在大规模模型中，这种优化可以显著减少训练和推理时间。
		- **性能表现**：尽管 RMSNorm 忽略了均值，但在许多实践中，它能够提供与 LayerNorm 相近甚至更好的性能，尤其是在某些特定任务或架构中。
		- **稳定性**：RMSNorm 通过仅依赖 RMS 进行规范化，可能在某些情况下提供更稳定的梯度流动，有助于训练过程的稳定性。
4. `预归一化(Pre-Norm)` 和 `后归一化(Post-Norm)`: `Qwen2DecoderLayer` 中有两个 RMSNorm 层: 
	- input_layernorm(`Qwen2RMSNorm`): 位于自注意力机制和 MLP 之前
	- post_attention_layernorm(`Qwen2RMSNorm`): 位于自注意力机制之后，进入 MLP 之前
	- Transformer 架构中的归一化层可以放置在不同的位置，主要有两种常见的设计：
		- **后归一化（Post-Norm）**：
		    - 归一化层位于子层（如自注意力层或 MLP 层）之后。
		    - 典型的 Transformer 论文如 "Attention is All You Need" 中采用此设计。
		    - 缺点：在非常深的模型中，可能导致梯度消失或梯度爆炸，影响训练稳定性。
		- **预归一化（Pre-Norm）**：
		    - 归一化层位于子层之前。
		    - 这种设计有助于缓解深层模型中的梯度问题，提高训练的稳定性和效率。
		    - 近年来，越来越多的研究和实践表明，预归一化在深层模型中表现更佳。
5. `强化位置信息`: 在进入 `lm_head` 之前再次应用 `rotary_emb`
6. `SiLU (Sigmoid Linear Unit)` 激活函数

### 示例流程

1. **输入处理**：
    - **文本输入**：提示或部分文本通过 `embed_tokens` 层进行标记化并转化为嵌入。
2. **模型推理**：
    - 嵌入传递到堆叠的解码层，逐层应用自注意力、前馈网络和归一化，生成上下文嵌入。
3. **输出生成**：
    - 最终嵌入通过 `lm_head` 转换为词汇表的 Logits。
    - 对 Logits 应用 Softmax 获取下一个 Token 的概率分布。
    - 选择概率最高的 Token（或使用采样策略如 top-k 或 nucleus 采样）生成下一个词。
4. **迭代生成**：
    - 将新生成的 Token 添加到输入序列，重复该过程，直到达到终止条件（例如序列结束 Token 或最大长度）。


## `Qwen2-VL-2B-Instruct` 模型结构

```scss
Qwen2VLForConditionalGeneration(
  (visual): Qwen2VisionTransformerPretrainedModel(
    (patch_embed): PatchEmbed(
      (proj): Conv3d(3, 1280, kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False)
    )
    (rotary_pos_emb): VisionRotaryEmbedding()
    (blocks): ModuleList(
      (0-31): 32 x Qwen2VLVisionBlock(
        (norm1): LayerNorm((1280,), eps=1e-06, elementwise_affine=True)
        (norm2): LayerNorm((1280,), eps=1e-06, elementwise_affine=True)
        (attn): VisionSdpaAttention(
          (qkv): Linear(in_features=1280, out_features=3840, bias=True)
          (proj): Linear(in_features=1280, out_features=1280, bias=True)
        )
        (mlp): VisionMlp(
          (fc1): Linear(in_features=1280, out_features=5120, bias=True)
          (act): QuickGELUActivation()
          (fc2): Linear(in_features=5120, out_features=1280, bias=True)
        )
      )
    )
    (merger): PatchMerger(
      (ln_q): LayerNorm((1280,), eps=1e-06, elementwise_affine=True)
      (mlp): Sequential(
        (0): Linear(in_features=5120, out_features=5120, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=5120, out_features=1536, bias=True)
      )
    )
  )
  (model): Qwen2VLModel(
    (embed_tokens): Embedding(151936, 1536)
    (layers): ModuleList(
      (0-27): 28 x Qwen2VLDecoderLayer(
        (self_attn): Qwen2VLSdpaAttention(
          (q_proj): Linear(in_features=1536, out_features=1536, bias=True)
          (k_proj): Linear(in_features=1536, out_features=256, bias=True)
          (v_proj): Linear(in_features=1536, out_features=256, bias=True)
          (o_proj): Linear(in_features=1536, out_features=1536, bias=False)
          (rotary_emb): Qwen2VLRotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=1536, out_features=8960, bias=False)
          (up_proj): Linear(in_features=1536, out_features=8960, bias=False)
          (down_proj): Linear(in_features=8960, out_features=1536, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((1536,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((1536,), eps=1e-06)
    (rotary_emb): Qwen2VLRotaryEmbedding()
  )
  (lm_head): Linear(in_features=1536, out_features=151936, bias=False)
)
```

由上 `Qwen2-VL-2B-Instruct` 模型主要由三个核心组件组成：

1. **视觉模块（`visual`）**：通过基于视觉 Transformer 的架构处理并编码视觉输入。
2. **语言模块（`model`）**：通过堆叠的解码层处理文本输入并生成输出。
3. **条件生成头（`lm_head`）**：将语言模块的输出转化为文本生成的词概率。

### 1. 视觉模块-值得注意的地方

1. `Conv3d`: 使用 3D 卷积覆盖非重叠的区域实现Patchify
	- 将输入的3个通道映射到1280个特征通道
	- 卷积核大小(kernel_size): `(2, 14, 14)`; 步幅(stride): `(2, 14, 14)`
	- `(N, 3, D, H, W) -> (N, 1280, D_out, H_out, W_out)`
		- 1280 是每个 Token 的嵌入维度（`embedding dimension`）。
		- **`D_out * H_out * W_out`** 表示生成的 Token 数量
		- 每个 Token 对应于输入图像中的一个 Patch
2. `VisionRotaryEmbedding`: 视觉特征添加位置信息
3. `LayerNorm`: 图像部分使用的是LN而非RMS, 但同样是Attn前后各一个(MLP 之前)
4. `QuickGELUActivation` 激活函数: 位于两个线性层之间，作为 MLP 的非线性激活函数
5. `PatchMerger`: 进行视觉token数的压缩与进一步提取特征(两层MLP):
	- **减少 Patch 数量**：合并相邻的 Patch，减少整体的 Token 数量
	- **增强特征表达和对齐维度**：两层MLP提取特征, 同时对齐语言模型维度
	- **GELU激活函数**

### 2. 语言模块-值得注意的地方

1. `词表大小`: 151657
2. `embed_matrix维度`: \[151,936, 1536\]
3. 其余注意事项 `同Qwen语言模型`

### 示例流程
1. **输入处理**：
    - **视觉输入**：通过视觉模块处理，将其转化为 Patch 嵌入并编码空间信息。
    - **文本输入**：通过语言模块的嵌入层将文本转化为特征向量。
2. **条件生成**：
    - 将视觉和文本信息整合到模型中，生成连贯且相关的输出。
3. **输出生成**：
    - `lm_head` 将语言模块的输出转化为词概率，生成最终文本。


qwen2vl 的一大创新就来源于对 `Patch` 的处理
### 详解一下 `PatchEmbed`

```python
class PatchEmbed(nn.Module):
	def __init__(
		self,
		patch_size: int = 14,
		temporal_patch_size: int = 2,
		in_channels: int = 3,
		embed_dim: int = 1152,
	) -> None:
		super().__init__()
		self.patch_size = patch_size
		self.temporal_patch_size = temporal_patch_size
		self.in_channels = in_channels
		self.embed_dim = embed_dim
		  
		kernel_size = [temporal_patch_size, patch_size, patch_size]
		self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)
	  
	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		target_dtype = self.proj.weight.dtype
		hidden_states = hidden_states.view(
		-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
		)
		hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
		return hidden_states
```

- 输入时 `hidden_states` 维度为: \[tokens=5704, dim=1176\]
	- 想读懂qwen2vl是怎么处理图像视频数据的, 必须搞明白 `processor` 源码是如何处理的, 尤其是这个 `hidden_states` 维度
	- 维度详情为: `(grid_t * grid_h * grid_w, channel * self.temporal_patch_size * self.patch_size * self.patch_size)`
	- 可以视作确定了 tokens个数, 并且确定了后续 3D卷积 处理patch
		- 相当于后面的 3D卷积 只针对一个 patch 进行, 卷出来之后 `时间步`, `长`, `宽` 维度直接为降为1

- `hidden_states = hidden_states.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)`
	- 又将 `hidden_states` 后面一个维度拆回去

- `hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)`
	- 这是一个 `dim=1176 -> self.embed_dim=1280` 过程
	- 其中 `self.proj` 是一个 3D 卷积: 
		- in_channels=3
		- embed_dim=1280
		- kernel_size=\[temporal_patch_size, patch_size, patch_size\]
		- stride=\[temporal_patch_size, patch_size, patch_size\]
		- bias=False
	- \>\>\> hidden_states.to(dtype=target_dtype).shape
		- torch.Size(\[5704, 3, 2, 14, 14\])
	- \>\>\> self.proj(hidden_states.to(dtype=target_dtype)).shape
		- torch.Size(\[5704, 1280, 1, 1, 1\])
	- \>\>\> self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim).shape
		- torch.Size(\[5704, 1280\])

### 详解一下 `PatchMerger`

```python
class PatchMerger(nn.Module):

	def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
	super().__init__()
	self.hidden_size = context_dim * (spatial_merge_size**2)
	self.ln_q = LayerNorm(context_dim, eps=1e-6)
	self.mlp = nn.Sequential(
		nn.Linear(self.hidden_size, self.hidden_size),
		nn.GELU(),
		nn.Linear(self.hidden_size, dim),
	)

def forward(self, x: torch.Tensor) -> torch.Tensor:
	x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
	return x
```

问题: 如果仔细阅读Qwen2VL的autoprocessor部分源码的话, 你会发现:
- tokenizer: 正常对文本部分进行分词, 使用"<|vision_start|><|image_pad|><|vision_end|>"来进行初步的视频tokens记录
- ImageProcessor: 按照 `时间步: 2, 长宽: 14x14` patchify
- 最终输出的inputs_id: 会将 "<|image_pad|>"等视觉pad变长, 但是实际长度却是 patchify 之后的 1/4, 这个原因就是来自于 `PatchMerger` 模块

解答: `PatchMerger` 类用于将多个 patch 合并成一个更高维度的表示。这种合并操作会显著减少 patch 的数量。具体来说，`PatchMerger` 通过 `spatial_merge_size`（默认为 2）将相邻的 patch 合并(`十字相邻`)。例如，`spatial_merge_size=2` 表示每 2x2 的 patch 会被合并为一个新的 patch。因此，原本的 patch 数量会减少为原来的 1/4。
- 重点在 `self.ln_q(x).view(-1, self.hidden_size)` 这行代码


# 配置文件

## 视觉-语言(Vision-Language, VL)模型中有多个配置文件

1. `config.json`
	`config.json` 在模型初始化时被加载, 模型的主要配置文件，用于定义模型的架构和参数。它包含了模型的结构信息，使得模型能够根据这些配置正确地初始化和运行。
```json
{
	"architectures": [
		"Qwen2VLForConditionalGeneration"
	],
	"attention_dropout": 0.0,
	"bos_token_id": 151643,
	"eos_token_id": 151645,
	"vision_start_token_id": 151652,
	"vision_end_token_id": 151653,
	"vision_token_id": 151654,
	"image_token_id": 151655,
	"video_token_id": 151656,
	"hidden_act": "silu",
	"hidden_size": 1536,
	"initializer_range": 0.02,
	"intermediate_size": 8960,
	"max_position_embeddings": 32768,
	"max_window_layers": 28,
	"model_type": "qwen2_vl",
	"num_attention_heads": 12,
	"num_hidden_layers": 28,
	"num_key_value_heads": 2,
	"rms_norm_eps": 1e-06,
	"rope_theta": 1000000.0,
	"sliding_window": 32768,
	"tie_word_embeddings": true,
	"torch_dtype": "bfloat16",
	"transformers_version": "4.41.2",
	"use_cache": true,
	"use_sliding_window": false,
	"vision_config": {
		"depth": 32,
		"embed_dim": 1280,
		"mlp_ratio": 4,
		"num_heads": 16,
		"in_chans": 3,
		"hidden_size": 1536,
		"patch_size": 14,
		"spatial_merge_size": 2,
		"spatial_patch_size": 14,
		"temporal_patch_size": 2
	},
	"rope_scaling": {
		"type": "mrope",
		"mrope_section": [
		16,
		24,
		24
		]
	},
	"vocab_size": 151936
}
```

`Qwen2VLConfig` 与 `LlavaConfig` 的初始化配置类略有不同
- `LlavaConfig` 类可以单独接收 `vision_config`, `text_config` 
- `Qwen2VLConfig` 类主要接受语言模型的配置参数，并通过 `vision_config` 参数嵌套包含视觉模型的配置, 可以直接传入json

`Qwen2VLConfig` 接收参数:
```python
class Qwen2VLConfig(PretrainedConfig):
	model_type = "qwen2_vl"
	keys_to_ignore_at_inference = ["past_key_values"]

	def __init__(
		self,
		vocab_size=152064,
		hidden_size=8192,
		intermediate_size=29568,
		num_hidden_layers=80,
		num_attention_heads=64,
		num_key_value_heads=8,
		hidden_act="silu",
		max_position_embeddings=32768,
		initializer_range=0.02,
		rms_norm_eps=1e-05,
		use_cache=True,
		tie_word_embeddings=False,
		rope_theta=1000000.0,
		use_sliding_window=False,
		sliding_window=4096,
		max_window_layers=80,
		attention_dropout=0.0,
		vision_config=None,
		rope_scaling=None,
		**kwargs,
	):
```

`Qwen2VLConfig` 官方示例: 
```python
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLConfig

# Initializing a Qwen2VL style configuration
configuration = Qwen2VLConfig()

# Initializing a model from the Qwen2-VL-7B style configuration
model = Qwen2VLForConditionalGeneration(configuration)

# Accessing the model configuration
configuration = model.config
```

自定义配置: 
```python
import json
from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration

# 读取 JSON 配置文件
with open("path/to/your/config.json", "r") as f:
    config_dict = json.load(f)

# 创建 Qwen2VLConfig 实例
qwen2vl_config = Qwen2VLConfig(**config_dict)

# 初始化并加载 Qwen2VL 模型
model = Qwen2VLForConditionalGeneration(qwen2vl_config)

```

2. `generation_config.json`
	`generation_config.json` 在调用生成方法（如 `generate()`）时被加载, 专门用于定义文本生成过程中的超参数和策略。这些配置项控制生成文本的行为，如生成长度、采样策略、温度、束搜索等, 例如:
	- **生成长度**：如最大生成长度（`max_length`）、最小生成长度（`min_length`）等。
	- **生成策略**：
	    - **采样相关**：如温度（`temperature`）、顶部K采样（`top_k`）、顶部P采样（`top_p`）等。
	    - **束搜索**：束宽度（`num_beams`）、束惩罚因子（`repetition_penalty`）等。
	- **其他生成参数**：如是否使用核采样（`do_sample`）、停止标记（`eos_token_id`）等。
```json
{
	"bos_token_id": 151643,
	"pad_token_id": 151643,
	"do_sample": true,
	"eos_token_id": [
		151645,
		151643
	],
	"repetition_penalty": 1.0,
	"temperature": 0.01,
	"top_p": 0.001,
	"top_k": 1,
	"transformers_version": "4.37.0"
}
```

值得注意的一个地方是, 在仅使用 `qwen2vl_config = Qwen2VLConfig(**config_dict)` 也就是 config.json 初始化模型的时候, 模型也会有推理参数, 这是 huggingface 源码中 PretrainedConfig 类初始化的时候会给一个默认的参数字典:
```python
@staticmethod
def _get_global_generation_defaults() -> Dict[str, Any]:
	return {
		"max_length": 20,
		"min_length": 0,
		"do_sample": False,
		"early_stopping": False,
		"num_beams": 1,
		"num_beam_groups": 1,
		"diversity_penalty": 0.0,
		"temperature": 1.0,
		"top_k": 50,
		"top_p": 1.0,
		"typical_p": 1.0,
		"repetition_penalty": 1.0,
		"length_penalty": 1.0,
		"no_repeat_ngram_size": 0,
		"encoder_no_repeat_ngram_size": 0,
		"bad_words_ids": None,
		"num_return_sequences": 1,
		"output_scores": False,
		"return_dict_in_generate": False,
		"forced_bos_token_id": None,
		"forced_eos_token_id": None,
		"remove_invalid_values": False,
		"exponential_decay_length_penalty": None,
		"suppress_tokens": None,
		"begin_suppress_tokens": None,
	}
```

3. `vocab.json`
	`vocab.json` 文件主要用于定义分词器的词汇表。它包含了模型可以识别和处理的所有词汇（tokens）及其对应的唯一标识符（IDs）。一些特殊tokens标记一般不会出现在这里    

4. `tokenizer_config.json`
	`tokenizer_config.json` 文件用于存储分词器的高层配置参数。这些参数影响分词器的行为和处理方式, 如填充方式(`padding_side`)、添加特殊标记(`add_special_tokens`)、最大序列长度(`model_max_length`)等. 但不涉及具体的词汇映射或分词逻辑:
```scss
{
	"add_prefix_space": false,
	"added_tokens_decoder": {
		"151643": {
			"content": "<|endoftext|>",
			"lstrip": false,
			"normalized": false,
			"rstrip": false,
			"single_word": false,
			"special": true
		},
		"151644": {
			"content": "<|im_start|>",
			"lstrip": false,
			"normalized": false,
			"rstrip": false,
			"single_word": false,
			"special": true
		},
		"151645": {
			"content": "<|im_end|>",
			"lstrip": false,
			"normalized": false,
			"rstrip": false,
			"single_word": false,
			"special": true
		},
		"151646": {
			"content": "<|object_ref_start|>",
			"lstrip": false,
			"normalized": false,
			"rstrip": false,
			"single_word": false,
			"special": true
		},
		"151647": {
			"content": "<|object_ref_end|>",
			"lstrip": false,
			"normalized": false,
			"rstrip": false,
			"single_word": false,
			"special": true
		},
		"151648": {
			"content": "<|box_start|>",
			"lstrip": false,
			"normalized": false,
			"rstrip": false,
			"single_word": false,
			"special": true
		},
		"151649": {
			"content": "<|box_end|>",
			"lstrip": false,
			"normalized": false,
			"rstrip": false,
			"single_word": false,
			"special": true
		},
		"151650": {
			"content": "<|quad_start|>",
			"lstrip": false,
			"normalized": false,
			"rstrip": false,
			"single_word": false,
			"special": true
		},
		"151651": {
			"content": "<|quad_end|>",
			"lstrip": false,
			"normalized": false,
			"rstrip": false,
			"single_word": false,
			"special": true
		},
		"151652": {
			"content": "<|vision_start|>",
			"lstrip": false,
			"normalized": false,
			"rstrip": false,
			"single_word": false,
			"special": true
		},
		"151653": {
			"content": "<|vision_end|>",
			"lstrip": false,
			"normalized": false,
			"rstrip": false,
			"single_word": false,
			"special": true
		},
		"151654": {
			"content": "<|vision_pad|>",
			"lstrip": false,
			"normalized": false,
			"rstrip": false,
			"single_word": false,
			"special": true
		},
		"151655": {
			"content": "<|image_pad|>",
			"lstrip": false,
			"normalized": false,
			"rstrip": false,
			"single_word": false,
			"special": true
		},
		"151656": {
			"content": "<|video_pad|>",
			"lstrip": false,
			"normalized": false,
			"rstrip": false,
			"single_word": false,
			"special": true
		}
	},
	"additional_special_tokens": ["<|im_start|>", "<|im_end|>", "<|object_ref_start|>","<|object_ref_end|>","<|box_start|>","<|box_end|>","<|quad_start|>","<|quad_end|>","<|vision_start|>","<|vision_end|>","<|vision_pad|>","<|image_pad|>","<|video_pad|>"],
	"bos_token": null,
	"chat_template": "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
	"clean_up_tokenization_spaces": false,
	"eos_token": "<|im_end|>",
	"padding_side": "left",
	"errors": "replace",
	"model_max_length": 32768,
	"pad_token": "<|endoftext|>",
	"split_special_tokens": false,
	"tokenizer_class": "Qwen2Tokenizer",
	"unk_token": null
}
```

5. `tokenizer.json`
`tokenizer.json` 是一个综合性文件，通常包含了分词器的完整配置和分词逻辑。它不仅包含 `vocab.json` 和 `tokenizer_config.json` 的内容(但`tokenizer.json`中的add_tokens可能没有`tokenizer_config.json`中全)，还包括分词器的具体实现细节，如分词合并规则、正则表达式等, 结合 `tokenizer_config.json` 的内容，提供完整的分词器配置

### `vocab.json`, `tokenizer.json` 和 `tokenizer_config.json`

- **`vocab.json` 与 `tokenizer.json`**：
    - `vocab.json` 提供了词汇到ID的基础映射，是分词器不可或缺的一部分。
    - `tokenizer.json` 将 `vocab.json` 嵌入其中，并结合分词规则（如BPE的合并规则）和行为参数，形成一个完整的分词器定义。
- **`tokenizer_config.json` 与 `tokenizer.json`**：
    - `tokenizer_config.json` 专注于高层次的分词器配置参数，控制分词器的整体行为。
    - `tokenizer.json` 不仅包含 `tokenizer_config.json` 的内容，还包括具体的分词逻辑和词汇表，是一个更全面的配置文件。

## `Qwen2-1.5B` config

```scss
Qwen2Config {
  "_attn_implementation_autoset": true,
  "_name_or_path": "/mnt/nas/ianli/models/Qwen2-1.5B",
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151643,
  "hidden_act": "silu",
  "hidden_size": 1536,
  "initializer_range": 0.02,
  "intermediate_size": 8960,
  "max_position_embeddings": 131072,
  "max_window_layers": 28,
  "model_type": "qwen2",
  "num_attention_heads": 12,
  "num_hidden_layers": 28,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.46.3",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
```