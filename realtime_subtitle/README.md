# reSpeaker 实时双语字幕

基于 reSpeaker XVF3800 麦克风阵列的本地实时双语字幕系统。硬件完成 AEC/波束成形/DOA/硬件 VAD，主机软件完成门控、分段、ASR 与翻译，浏览器实时显示双语字幕。

## 快速开始

```bash
conda activate funasr
cd realtime_subtitle

# 默认：FunASR 流式 2-pass（说话期间即出 partial，整段音频 480ms 一步）
python main.py

# 关闭流式，仅离线整句解码
python main.py --no-streaming

# 流式粒度：480ms（默认，延迟低）/ 600ms（上游默认，前瞻更长）
python main.py --asr-chunk-size 600

# Whisper 兜底引擎（不支持流式，自动走离线整句）
python main.py --asr whisper --model tiny --port 8765

# 门控模式
python main.py --vad-gate hw|and|confirm   # 默认 and

# 波束侦察/路由（只读探测优先）
python main.py --beam-recon
python main.py --beam-select
python main.py --beam-channels        # 两个可编程输出落在哪个采集通道

# 双轨：两个固定波束各一条字幕通道（两人各占一轨）
python main.py --beam-lanes 180 0     # 波束指向（度）；不传即单轨
python main.py --beam-lanes 180 0 --beam-no-route   # 只采集不分流，不动设备

# 列出音频设备
python main.py --list-devices
```

浏览器打开 <http://127.0.0.1:8765/>。

## 架构一览

- **数据面**：USB Audio 6ch → `audio_callback` 按 lane 切分 → 各 lane 的 `audio_queue` → 主循环
- **控制面**：USB vendor transfer → `VadDoaReader`（隔离 RETRY 长阻塞）→ `vad_queue`
- **三层门控**：硬件 VAD ∧ Silero 软件 VAD ∧ flush 段级时长/占比/能量过滤
- **推理流水线**：`TranscriberWorker`（ASR）→ 立即广播 `source(id)`；`TranslationThread`（NLLB ct2-int8）→ 同 id 补推 `translation`，翻译不阻塞原文
- **前端**：单文件 `web/index.html`（后端仅服务 `/` 与 `/index.html`），partial 即时整段替换（无逐字动画）、final 整段替换、discard 删除、按 id 合并

延迟统计只打印在后端终端（`[首字]` / `[耗时]`），前端不接收 latency 字段。

### 采集通道映射

采集流恒为 6 通道，路由选择只决定**前两个**通道装什么，通道 2–5 是固定的四路原始麦克风：

| 通道 | `--audio-source` | 内容 |
|---|---|---|
| ch0 | `conference`（默认） | 处理输出（AEC+波束+后处理） |
| ch1 | `asr` | ASR 波束输出（auto-select，无后处理） |
| ch2–ch5 | `mic0`–`mic3` | 四路原始麦克风 |

### 双轨模式

`--beam-lanes AZ1 AZ2` 启用两条字幕通道：`OP_L` 取波束 1、`OP_R` 取波束 2（实测确认 `(7,0)`/`(7,1)` 即 beam1/beam2，故无需扫描），两波束指向给定方位、**不开启能量门控**（门控会让一个人说话时静音另一条 lane）。每轨用自己的 `SileroVad`、`Segmenter`、音频队列与 DOA 统计；门控默认切到 `energy`（按 `SPENERGY[beam]` 分轨，硬件标志是设备级、无法分轨）。

未传 `--beam-lanes` 时行为与引入双轨前**完全一致**。退出时路由与波束指向都会恢复，全程不写 flash。

详细设计见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## USB 权限

```bash
sudo cp 99-respeaker-xvf3800.rules /etc/udev/rules.d/
sudo udevadm control --reload && sudo udevadm trigger
```

## 模型

- ASR：FunASR `paraformer-zh` + `ct-punc`（流式另需 `paraformer-zh-streaming`），首次运行自动下载
- 翻译：NLLB-200 CT2 INT8，优先 `models/nllb-200-distilled-1.3B-ct2-int8`，未下载时自动回退 HF 缓存中的 600M
