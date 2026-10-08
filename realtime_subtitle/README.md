# reSpeaker 实时双语字幕

基于 reSpeaker XVF3800 麦克风阵列的本地实时双语字幕系统。硬件完成 AEC/波束成形/DOA/硬件 VAD，主机软件完成门控、分段、ASR 与翻译，浏览器实时显示双语字幕。

## 快速开始

```bash
conda activate funasr          # x86 开发机；Jetson 上用 venv，见下节
cd realtime_subtitle

# 默认：FunASR 流式 2-pass（说话期间即出 partial，整段音频 480ms 一步）
python main.py

# 流式粒度：480ms（默认，延迟低）/ 600ms（上游默认，前瞻更长）
python main.py --asr-chunk-size 600
```

也可以直接用 `./run.sh`（自动激活 venv 或 conda，并以 `python -u` 启动）。

输入设备与采集通道**不可配置**：设备按名字自动识别 XVF3800（识别不到则回落系统默认并告警），采集固定取索引 0，即"处理输出"（AEC+波束+后处理完整链）。需要换通道时改 `main.py` 的 `CAPTURE_CHANNEL`。

浏览器打开 <http://127.0.0.1:8765/>；从其他机器访问用本机 IP，例如 <http://192.168.5.21:8765/>（默认已绑定 `0.0.0.0`，可用 `--host` 改）。

### 推理设备

启动横幅第一行会写明实际生效的后端，无需猜测：

```
[compute] aarch64/linux | torch 2.13.0+cu130 (13.0) | GPU Orin | ctranslate2 4.8.2 CPU-only
[ASR] FunASR Paraformer (zh, streaming 2pass, stride 480ms, device cuda)
[NLLB] model: ... (cpu, int8)
[VAD] Silero software VAD ready (device cpu)
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--device-type` | `auto` | ASR 设备。`auto` = 有可用 CUDA 就用，否则 CPU |
| `--nllb-device` | 同 `--device-type` | 仅翻译。aarch64 上 PyPI 的 ctranslate2 是纯 CPU 构建，会自动回落到 CPU `int8` 并打警告，而不是启动失败 |
| `--nllb-compute-type` | 随设备 | 手动钉住量化类型（`int8` / `int8_float16` / `float16`）。**GPU 上不要用 int8**：与 ASR 并发时只有 6/24 成功，见 [ARCHITECTURE.md](ARCHITECTURE.md) §7.1.1 |
| `--nllb-beam` | `1` | NLLB beam 宽度。贪心比 4 快约 20%（x86 246→198ms，Orin 1026→828ms）；译文读着别扭时可调到 4 |
| `--no-hw-vad` | 关 | 仅用软件 VAD 门控。**控制面不可用时必须加**，否则硬件标志恒 0、门控恒假，表现为“跑起来但永远不出字幕” |

Silero VAD 固定跑在 CPU（无 CLI 入口）：模型极小但每秒调用 30 次，GPU 启动开销大于收益，还会与 ASR 争显存。

模型权重本身与平台无关（CT2 int8 / Paraformer state_dict 都不带 CPU/GPU 标志），差别只在引擎：torch 必须是 CUDA 构建、ctranslate2 必须编译了 CUDA。

Orin 上翻译比 x86 慢约 4 倍（每句 ~0.8s vs ~0.2s），原因是 PyPI 的 aarch64 ctranslate2 走 **Ruy** 的 int8 kernel，而 x86 轮子用 MKL/oneDNN；与 CPU 频率无关。

**已解决**：按 [requirements-jetson.txt](requirements-jetson.txt) §3 在目标机编译带 CUDA 的 ctranslate2 后，翻译走 GPU，Orin 上 **828ms → 346ms**（ASR 并发下中位；零失败）。量化类型选 `float16` 而非 `int8`，原因是后者与 ASR 并发时只有 6/24 成功 —— 选型依据与实测见 [ARCHITECTURE.md](ARCHITECTURE.md) §7.1.1，运行时注意事项见 [requirements-jetson.txt](requirements-jetson.txt) §3b。

### Jetson / aarch64 部署

```bash
sudo apt-get install -y python3-pip python3-venv libportaudio2   # pip/venv/PortAudio 都非 pip 可提供
python3 -m venv ~/venvs/subtitle && . ~/venvs/subtitle/bin/activate

# torch/torchaudio 必须是 CUDA 构建，否则 torch.cuda.is_available() 为 False，全部模型静默跑 CPU
pip install --extra-index-url https://download.pytorch.org/whl/cu130 \
    "torch==2.13.0+cu130" "torchaudio==2.11.0+cu130"

# 其余依赖用约束文件防止装成 CPU 版 torch
pip install -c <(echo -e "torch==2.13.0+cu130\ntorchaudio==2.11.0+cu130") -r requirements.txt

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"   # 末项必须 True
```

> 首次调用 CUDA 时 torch 会警告“Orin which is of compute capability (CC) 8.7 ...
> not support this GPU”。**这是 PyTorch 官方轮子的普遍情况，可忽略**：aarch64 与 x86
> 轮子的 `get_arch_list()` 都不含 `sm_87`（Orin 专属目标），实测靠 PTX JIT 回退，
> 数值正确（相对误差 8.9e-07）、首次多耗几十毫秒。详见
> [requirements-jetson.txt](requirements-jetson.txt) §2b。

细节（轮子索引、ct2 源码编译上 GPU、HF/ModelScope 缓存搬运的两个坑）见 [requirements-jetson.txt](requirements-jetson.txt)。

## 架构一览

- **数据面**：USB Audio 6ch → `audio_callback` 取出选定通道 → `audio_queue` → 主循环
- **控制面**：USB vendor transfer → `VadDoaReader`（隔离 RETRY 长阻塞）→ `vad_queue`
- **三层门控**：硬件 VAD ∧ Silero 软件 VAD ∧ flush 段级时长/占比/能量过滤
- **推理流水线**：`TranscriberWorker`（ASR）→ 立即广播 `source(id)`；`TranslationThread`（NLLB ct2-int8）→ 同 id 补推 `translation`，翻译不阻塞原文
- **前端**：单文件 `web/index.html`（后端仅服务 `/` 与 `/index.html`），partial 即时整段替换（无逐字动画）、final 整段替换、discard 删除、按 id 合并
- **设备选择**：`compute.py` 集中所有平台相关决策（设备探测、量化类型、平台横幅），各 provider 不再自行猜测
- **设置**：翻译方向改动即生效（`onchange`），并存入 `runtime_settings.json` 以便重启后保留；响应了抽屉上“运行时配置立即生效”的承诺

延迟统计只打印在后端终端（`[首字]` / `[耗时]`），前端不接收 latency 字段。

### 采集通道映射

采集流恒为 6 通道，程序固定取**索引 0（横幅 ch1）**，即"处理输出"（AEC+波束+后处理完整链）。其余通道不提供 CLI 入口，仅在此记录各自位置：

| 索引 | 横幅 | 内容 |
|---|---|---|
| 0 | ch1 | 处理输出（AEC+波束+后处理）—— **程序使用的就是这个** |
| 1 | ch2 | ASR 波束输出（auto-select，无后处理） |
| 2–5 | ch3–ch6 | 四路原始麦克风，每麦一路 |

这四个参数（`--audio-source`/`--ch`/`--in-channels`）的唯一作用就是选一个索引，因此已移除；需要换通道改 `main.py` 的 `CAPTURE_CHANNEL`。四路原始麦克风是固定的每麦一路，给它们起名字不比索引多出能力。

波束（转向/路由）**不再由程序配置**，理由和实测数据见 [ARCHITECTURE.md](ARCHITECTURE.md) §8。该能力**不再有任何界面入口**：前端「USB 命令」面板与其后端命令已移除；设备层的 `usb_device.read()/write()` 与 flash 守卫仍保留，供将来需要时使用。

详细设计见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## USB 权限

```bash
sudo cp 99-respeaker-xvf3800.rules /etc/udev/rules.d/
sudo udevadm control --reload && sudo udevadm trigger
```

非 root 用户还需能访问裸 USB 与音频设备，通常要把用户加入 `plugdev` / `audio` 组（重新登录后生效）：

```bash
sudo usermod -aG plugdev,audio $USER
```

规则未生效时 `ReSpeaker()` 会抛 “reSpeaker XVF3800 not found (VID 0x2886 / PID 0x001A)”（能枚举设备但读参数会 `Access denied`），程序继续运行但门控永不开——此时加 `--no-hw-vad` 可先只靠软件 VAD 跑起来。

### 用自播做冒烟测试时的注意点

拿 `aplay` 播一段人声到阵列自己的输出通道，是无人说话时最方便的激励源，但**固件 AEC 会把自播当作远端参考**，因此：

- 各通道能量只上升 1.1～1.6 倍（实测），远不像外部声源
- 输出音量默认 62%（-23dB）时，`DOA_VALUE` 的语音标志**全程为 0**，门控不会开
- 把 `amixer -c <card> sset 'PCM' 100%` 拉满后，标志才会间歇为 1（实测 11/30 次）

也就是说自播能触发管线，但得到的段普遍太短，会被第三层段级过滤器按 `duration` 拒掉（日志尾部会写 `[seg] 交付 0, 丢弃 N [duration×N]`）。**要验证真实的识别质量，请用外部声源或直接对人说话。**

## 模型

- ASR：FunASR `paraformer-zh` + `ct-punc`（流式另需 `paraformer-zh-streaming`），首次运行自动下载到 `~/.cache/modelscope/models/`
- 翻译：NLLB-200 CT2 INT8，优先 `models/nllb-200-distilled-1.3B-ct2-int8`，未下载时自动回退 HF 缓存中的 600M
- 可用 `NLLB_MODEL_DIR=/path/to/ct2-model` 直接指定翻译模型目录，跳过目录探测（也免去在目标机上复刻 HF 缓存布局）
