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
② 软件 VAD   Silero torch jit，主循环内联，512 帧 + carry
③ 段级过滤   flush 时 O(1)：时长 → 占比 → 能量
```

模式 `--vad-gate hw|and|confirm`（默认 and）。`and` 每拍合取；`confirm` 硬件开门后由软件连续确认/否决，两套状态机独立实现。

关键语义：

- Silero `reset()` 只在段真正结束后调用（段边沿清零会冷启动 RNN 误杀开段）
- 占比分母只统计到最后一个有声 tick（句尾静音等待期不计入，防短句被稀释）
- 时长只累加门开期间的样本（有声时长）：尾静音等待期约 0.6s，若计入则 `min_seconds` 形同虚设、短句豁免的 1s 阈值也会错位
- 短句豁免仅对 <1s 且峰值概率 ≥0.5 的段生效（纯 avg_prob 豁免会放行 AGC 底噪）
- 所有丢弃统一走 `discard_segment()`：stream_abort + 广播 discard + 清缓冲 + 复位 + 回 IDLE（防 partial 残留）

## 3. 分段状态机

```
IDLE → SPEAKING → TRAILING_SILENCE → FLUSH ─┬→ DISCARD（统一丢弃路径）
                                             └→ SUBMIT_ASR
```

- 音频三归属：utterance（正文）/ pre_roll（1.0s 环形缓冲，开段时整体作 seed 喂给流式模型）/ 丢弃
- pre_roll 无条件喂入（含静音），保证开段瞬间窗口已满；flush/discard 都清空它，否则尾部语音会让下一段流重复解码出可见重复文本
- 参数：`vad_interval=0.1s`、`silence_hold=0.6s`、`min_seconds=0.3`、`max_seconds=20`、`pre_roll=1.0s`
- 超长段强制切分；DOA 众数随段保存（仅展示与波束参考，不参与过滤）

## 4. 推理层

| 执行体 | 位置 | 模型 |
|---|---|---|
| sw VAD | 主循环内联 | Silero torch jit（官方 silero-vad 包 / 本地 silero_vad.jit） |
| ASR | TranscriberWorker | FunASR paraformer-zh + ct-punc（主）/ faster-whisper（兜底） |
| 翻译 | TranslationThread | NLLB-200-distilled-600M ct2-int8（1.3B 目录存在时优先） |

FunASR 流式 2-pass 契约：流式 generate 返回**增量文本**，provider 累计拼接为 partial；`finish_online()` 用 `is_final=True` 冲出尾字并立即广播，整段音频交给 `FinalDecodeThread` 离线重解出 final 替换 partial。

流式解码参数：`chunk_size` 默认 `[5,8,4]`（480ms stride，`--asr-chunk-size 600` 可换回上游默认 `[5,10,5]`），并显式传 `encoder_chunk_look_back=4` / `decoder_chunk_look_back=1`（仅在 cache 首次初始化时生效）。provider 只在累积文本增长时返回 non-empty，避免同一 stride 内重复广播相同 partial。

语言路由：源语言由**识别文本的字形**判定（含假名→日语，含谚文→韩语，含汉字→中文，其余归为英语），不取 ASR 的 `language` 字段——FunASR 恒报 `zh`，那是"本模型输出中文"而非"听出的是中文"，若采信它则英语输入会走中→外分支，"其他语言→中文"形同虚设。非中文一律走 `other_to` 方向，未知源语种也照样翻译而非静默跳过。解码 beam=4 + length_penalty=1.0 + repetition_penalty=1.1；短段（<2 字符）不送翻译。

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
出站  source{id,ts,doa,text,lang}
      partial{id,ts,text}
      final{id,ts,text,lang}
      discard{id}
      translation{id,ts,text,target_lang}
      vad.state{hw_vad,sw_prob,gate_mode,segment_state}
      cmd_result{ok,data|error}
入站  cmd: list_commands / read / write / get_target_langs / set_target_langs / get_zones / set_zones
```

延迟测量只留在后端终端（`[首字]` / `[耗时]` 行 + 退出汇总），前端是纯渲染器，不接收任何 latency 字段。

## 6. 前端

单文件 `web/index.html`（后端只有 `/` 与 `/index.html` 路由，CSS/JS 全内联）。深色玻璃拟态；字幕状态机 partial 即时整段替换（后端发的是累积文本）→ final 整段替换 → discard 删除 → translation 填充；无逐字动画，每条消息到达即刻上屏。卡片上限 50 条；断线指数退避重连；XSS 全量转义。

## 7. 波束路由

`BeamRouter`：`recon()` 只读探测 → `describe()` 检测映射歧义（歧义时拒绝写入）→ `apply()` 先存原路由、写后验证、失败 `restore()` 回滚。门控/推理通道与候选波束通道相互独立。

## 8. 延迟预算（基线）

```
译文端到端 ~1.0s = silence_hold 0.6s(~60%) + 轮量化 ~10% + ASR 0.15~0.25s + NLLB 0.2~0.3s + WS <20ms
```

推理仅占 ~0.4s，优化重点在切句策略（软件 VAD 连续低概率帧提前 flush）而非模型。翻译（600M）增量 ~0.2–0.4s，因与 source 广播解耦，不阻塞原文显示。
