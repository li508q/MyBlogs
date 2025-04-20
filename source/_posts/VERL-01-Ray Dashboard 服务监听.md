---
title: VERL-01-Ray Dashboard 服务监听
date: 2025-04-20 17:15:00
updated: 2025-04-20 17:15:00
tags: VERL; Ray
categories: 框架
cover: /img/VolcanoEngine.png
top_img: /img/VolcanoEngine_long.png
---

## 任务背景

我的任务背景是: 在远程服务器（IP: `REMOTE_SERVER_IP`）上利用火山引擎的 VERL 框架进行 PPO 算法训练。为此，我使用了官方文档中提供的镜像，并在其中源码安装了 VERL。 而在使用 Docker 容器化部署复杂的分布式应用时, 网络配置往往是一个挑战, 在远程服务器上启动 Docker Container 运行 VERL 强化学习框架（基于 Ray）的同时, 我们也想远程访问 Ray Dashboard, 本文记录了工作流. 

## 使用`host` 网络模式

Docker 默认使用 `bridge` 网络模式，为每个容器创建独立的网络命名空间，拥有自己的 IP 地址和端口空间，提供了良好的隔离性，但也引入了网络虚拟化开销。`--network=host` 模式则完全不同，它指示 Docker 放弃为容器创建隔离的网络命名空间，让容器直接共享宿主机的网络栈。这意味着：

1. **共享网络接口**: 容器内看到网络接口、IP 地址、路由表等与宿主机完全相同。
2. **端口直接监听**: 容器内进程监听的任何端口，都直接绑定在宿主机的对应端口上。例如，容器内监听 TCP 80 端口，效果等同于宿主机进程直接监听 TCP 80 端口。
3. **无需端口映射**: 由于共享网络栈，Docker 的 `-p host_port:container_port` 端口映射机制在此模式下无效且无必要。

```bash
docker run -it \
    --gpus all \
    --name ianli-verl \
    --network=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /path/to/ianli:/mnt/ianli \
    ianli/verl:v1.0 \
    bash
```

## 显式启动 Ray Head

下面, 我们的核心目标是在容器内运行 VERL 的 PPO 训练脚本 (`main_ppo.py`)，该脚本内部使用 Ray 进行分布式计算。过程中我希望在本地 Mac 通过 SSH 连接到服务器后，Dashboard 能够通过浏览器访问 `http://REMOTE_SERVER_IP:8265(启动时默认端口为8265)` 来实时监控 Ray Dashboard 上的训练状态。

我们在运行 main_ppo.py 脚本之前，在容器的 shell 中执行：

``` Bash
ray start --head --dashboard-host 0.0.0.0 -- ...
```
- `--head`: 声明此节点为 Ray 集群的头节点。
- `--dashboard-host 0.0.0.0`: 将 Ray Dashboard 绑定到 `0.0.0.0`，使其在宿主机的所有网络接口上监听。
- `-- ...`: Ray 集群还有很多可选服务和配置, 可参考其官方文档。

接下来, 我们可以运行相关训练脚本, 例如 verl 框架中 example: 

```Bash
python3 -m verl.trainer.main_ppo \
	<your_hydra_configs>
```

你也可以直接向 Ray 后台直接递交任务, Ray 集群会自动管理 pending, running等:

```Bash
ray job submit \
-- ... \
-- python3 -m verl.trainer.main_ppo \
	<your_hydra_configs>
```

如果不显示启动 Ray, 直接运行 verl 的训练脚本同样也可以正确启用ray, 因为在 main_ppo.py中:

```Python
# From verl/main_ppo.py
def run_ppo(config) -> None:
    #... (Environment setup)...
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN'
            }
        })

    runner = TaskRunner.remote() # Instantiate a Ray Actor
    ray.get(runner.run.remote(config)) # Run the main logic remotely
#... (TaskRunner class definition using @ray.remote)...

if __name__ == '__main__':
    main() # Calls run_ppo via hydra
```

这里会将 dashboard服务默认绑定到 `localhost`.

**验证监听** 我这里将服务绑定到 `0.0.0.0` 是因为绑定到默认的 `localhost` 不知道为什么在我目前使用的远程服务器上, 容器内的端口内容就是无法被我的远程宿主机监听到. 在这里我们可以在宿主机中验证一下我们的服务是否被正确的监听了: 

在容器内执行完 `ray start --head -- ...` 命令后，再次**回到宿主机终端**运行:

```
sudo ss -tlpn | grep 8265
# 或
sudo netstat -tulnp | grep 8265
```

查看是否有预期的监听信息. 

### SSH 端口转发

最后, 我们要将远程宿主机, 也就是我们的远程服务器上监听的服务转发到我们的本地机器.

1. **在本地机器 (例如, Mac) 上**，建立 SSH 连接，并将远程服务器的 `localhost:8265` 转发到本地的 `8265` 端口：

```Bash
ssh -L 8265:localhost:8265 ianli@REMOTE_SERVER_IP
```

2. **在本地机器的浏览器中**，访问 `http://localhost:8265`。
