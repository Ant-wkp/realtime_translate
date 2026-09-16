# 架构

一句话：XVF3800 硬件负责音频增强与空间感知，Python 单进程多线程流水线负责门控/分段/识别/翻译，浏览器负责显示与运行时控制。

## 1. 执行流（6 + 1）

| # | 执行流 | 类型 | 节拍 | 职责 |
|---|---|---|---|---|
| 1 | audio_callback | PortAudio 线程 | 按块 | 6ch 抽单通道，入 audio_queue |
| 2 | VadDoaReader | daemon | 100ms | USB 读 DOA_VALUE（角度+语音活动），入 vad_queue |
| 3 | 主循环 | 显式循环 | sleep(0.1) | 排空队列、门控裁决、分段状态机、结果派发 |
| 4 | TranscriberWorker | daemon | task_queue.get(0.5) | ASR 串行推理，保序 |
| 5 | TranslationThread | daemon | job_queue.get(0.5) | NLLB 串行翻译 |
| 6 | BroadcastServer | asyncio | _pump 10ms | WebSocket 广播 |

主循环内所有重操作都是队列 put；USB 控制传输（RETRY 可达 1s+）被 VadDoaReader 隔离在控制面。

## 2. 三层门控

```
① 硬件 VAD   XVF3800 固件能量门，走控制面
② 软件 VAD   Silero ONNX，主循环内联，512 帧 + carry
③ 段级过滤   flush 时 O(1)：时长 → 占比 → 能量
```

模式 `--vad-gate hw|and|confirm`（默认 and）。`and` 每拍合取；`confirm` 硬件开门后由软件连续确认/否决，两套状态机独立实现。

关键语义：

- Silero `reset()` 只在段真正结束后调用（段边沿清零会冷启动 RNN 误杀开段）
- 占比分母只统计到 last_speech_at（句尾静音等待期不计入，防短句被稀释）
- 短句豁免仅对 <1s 且峰值概率 ≥0.5 的段生效（纯 avg_prob 豁免会放行 AGC 底噪）
- 所有丢弃统一走 `discard_segment()`：stream_abort + 广播 discard + 清缓冲 + 复位 + 回 IDLE（防 partial 残留）

## 3. 分段状态机

```
IDLE → SPEAKING → TRAILING_SILENCE → FLUSH ─┬→ DISCARD（统一丢弃路径）
                                             └→ SUBMIT_ASR
```

- 音频三归属：utterance（正文）/ pre_roll（~0.5s 建段时并入头部，防丢字）/ 丢弃
- 参数：`vad_interval=0.1s`、`silence_hold=0.6s`、`min_seconds=0.5`、`max_seconds=20`
- 超长段强制切分；DOA 众数随段保存（仅展示与波束参考，不参与过滤）

## 4. 推理层

| 执行体 | 位置 | 模型 |
|---|---|---|
| sw VAD | 主循环内联 | Silero v5 ONNX |
| ASR | TranscriberWorker | FunASR paraformer-zh + ct-punc（主）/ faster-whisper（兜底） |
| 翻译 | TranslationThread | NLLB-200-distilled-1.3B ct2-int8 |

FunASR 流式 2-pass 契约：流式 generate 返回**增量文本**，provider 累计拼接为 partial；finish 用完整音频离线重解出 final 替换 partial。

语言路由（高召回）：文本含任意 CJK 字符即判中文源；未知语种告警并跳过翻译显示原文，不静默回退。解码 beam=4 + length_penalty=1.0 + repetition_penalty=1.1；短段（<2 字符）不送翻译。

## 5. 数据流契约

原文先推、译文补推、按 id 合并：

```
flush 段(id) ──→ TranscriberWorker ──→ source(id) 立即广播
                        │
                        └→ TranslationThread ─→ translation(id) 补推
前端按 id 合并双语卡片，翻译不阻塞原文显示
```

消息协议：

```
出站  source{id,ts,doa,direction,text,lang}
      partial{id,ts,text}
      final{id,ts,text,lang}
      discard{id}
      translation{id,ts,text,target_lang,latency_ms}
      vad.state{hw_vad,sw_prob,gate_mode,segment_state}
      cmd_result{ok,data|error}
入站  cmd: list_commands / read / write / get_target_langs / set_target_langs / get_zones / set_zones
```

## 6. 前端

单文件 `web/index.html`（后端只有 `/` 与 `/index.html` 路由，CSS/JS 全内联）。深色玻璃拟态；字幕状态机 partial → final 替换 → discard 删除 → translation 填充；卡片上限 50 条；断线指数退避重连；XSS 全量转义。

## 7. 波束路由

`BeamRouter`：`recon()` 只读探测 → `describe()` 检测映射歧义（歧义时拒绝写入）→ `apply()` 先存原路由、写后验证、失败 `restore()` 回滚。门控/推理通道与候选波束通道相互独立。

## 8. 延迟预算（基线）

```
译文端到端 ~1.0s = silence_hold 0.6s(~60%) + 轮量化 ~10% + ASR 0.15~0.25s + NLLB 0.2~0.3s + WS <20ms
```

推理仅占 ~0.4s，优化重点在切句策略（软件 VAD 连续低概率帧提前 flush）而非模型。1.3B 翻译增量 ~0.5–0.9s，因与 source 广播解耦，不阻塞原文显示。
