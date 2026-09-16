# reSpeaker 实时双语字幕

基于 reSpeaker XVF3800 麦克风阵列的本地实时双语字幕系统。硬件完成 AEC/波束成形/DOA/硬件 VAD，主机软件完成门控、分段、ASR 与翻译，浏览器实时显示双语字幕。

## 快速开始

```bash
conda activate funasr
cd realtime_subtitle

# FunASR 中文主引擎（离线整句）
python main.py

# FunASR 流式 2-pass（说话期间即出 partial）
python main.py --streaming

# Whisper 兜底引擎
python main.py --asr whisper --model tiny --port 8765

# 门控模式
python main.py --vad-gate hw|and|confirm   # 默认 and

# 波束侦察/路由（只读探测优先）
python main.py --beam-recon
python main.py --beam-select

# 列出音频设备
python main.py --list-devices
```

浏览器打开 <http://127.0.0.1:8765/>。

## 架构一览

- **数据面**：USB Audio 6ch → `audio_callback` 抽单通道 → `audio_queue` → 主循环
- **控制面**：USB vendor transfer → `VadDoaReader`（隔离 RETRY 长阻塞）→ `vad_queue`
- **三层门控**：硬件 VAD ∧ Silero 软件 VAD ∧ flush 段级时长/占比/能量过滤
- **推理流水线**：`TranscriberWorker`（ASR）→ 立即广播 `source(id)`；`TranslationThread`（NLLB-1.3B ct2-int8）→ 同 id 补推 `translation`，翻译不阻塞原文
- **前端**：单文件 `web/index.html`（后端仅服务 `/` 与 `/index.html`），partial→final 替换、discard 删除、按 id 合并

详细设计见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## USB 权限

```bash
sudo cp 99-respeaker-xvf3800.rules /etc/udev/rules.d/
sudo udevadm control --reload && sudo udevadm trigger
```

## 模型

- ASR：FunASR `paraformer-zh` + `ct-punc`（流式另需 `paraformer-zh-streaming`），首次运行自动下载
- 翻译：`models/nllb-200-distilled-1.3B-ct2-int8`（CTranslate2 INT8；目录缺失时回退 600M）
