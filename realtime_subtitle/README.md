# reSpeaker 实时双语字幕

基于 reSpeaker XVF3800 麦克风阵列的本地实时双语字幕系统。硬件完成 AEC/波束成形/DOA/硬件 VAD，主机软件完成门控、分段、ASR 与翻译，浏览器实时显示双语字幕。

## 快速开始

```bash
conda activate funasr
cd realtime_subtitle

# 默认：FunASR 流式 2-pass（说话期间即出 partial，整段音频 480ms 一步）
python main.py

# 流式粒度：480ms（默认，延迟低）/ 600ms（上游默认，前瞻更长）
python main.py --asr-chunk-size 600

# 换处理输出或直指采集通道
python main.py --audio-source asr
python main.py --ch 2

# 列出音频设备
python main.py --list-devices
```

浏览器打开 <http://127.0.0.1:8765/>。

## 架构一览

- **数据面**：USB Audio 6ch → `audio_callback` 取出选定通道 → `audio_queue` → 主循环
- **控制面**：USB vendor transfer → `VadDoaReader`（隔离 RETRY 长阻塞）→ `vad_queue`
- **三层门控**：硬件 VAD ∧ Silero 软件 VAD ∧ flush 段级时长/占比/能量过滤
- **推理流水线**：`TranscriberWorker`（ASR）→ 立即广播 `source(id)`；`TranslationThread`（NLLB ct2-int8）→ 同 id 补推 `translation`，翻译不阻塞原文
- **前端**：单文件 `web/index.html`（后端仅服务 `/` 与 `/index.html`），partial 即时整段替换（无逐字动画）、final 整段替换、discard 删除、按 id 合并

延迟统计只打印在后端终端（`[首字]` / `[耗时]`），前端不接收 latency 字段。

### 采集通道映射

采集流恒为 6 通道，但只有两个处理输出有名字——它们背后是两套不同的 DSP 链；其余靠 `--ch` 索引直达：

| 通道 | `--audio-source` | 内容 |
|---|---|---|
| ch0 | `conference`（默认） | 处理输出（AEC+波束+后处理） |
| ch1 | `asr` | ASR 波束输出（auto-select，无后处理） |
| ch2–ch5 | `--ch 2..5` | 四路原始麦克风 |

不加 `--ch` 时只看 `--audio-source`；`--ch` 优先于它。四路原始麦克风是固定的每麦一路，再给它们起名字不比索引多出能力。

波束（转向/路由）**不再由程序配置**，理由和实测数据见 [ARCHITECTURE.md](ARCHITECTURE.md) §8。需要时可在前端「USB 命令」面板手动读写，会写 flash 的命令被 `usb_device.py` 拦住。

详细设计见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## USB 权限

```bash
sudo cp 99-respeaker-xvf3800.rules /etc/udev/rules.d/
sudo udevadm control --reload && sudo udevadm trigger
```

## 模型

- ASR：FunASR `paraformer-zh` + `ct-punc`（流式另需 `paraformer-zh-streaming`），首次运行自动下载
- 翻译：NLLB-200 CT2 INT8，优先 `models/nllb-200-distilled-1.3B-ct2-int8`，未下载时自动回退 HF 缓存中的 600M
