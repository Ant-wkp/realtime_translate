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
- 实现见 `segmenter.py`：`Segmenter` 持有单个 lane 的全部状态，`feed(chunks, gate_fn, hw_vad, now)` 每次 tick 返回 `Action` 列表（`open`/`feed`/`flush`/`discard`），由 `main.py` 的 `apply_actions()` 翻译成对转录器与前端广播的调用。软 VAD 概率在 `Segmenter` 内部计算一次，同时驱动门控判定、统计与句尾确认，避免同一个 tick 出现多个读数
- 该文件是纯逻辑、无 I/O，可脱离硬件测试（`tests/test_segmenter.py`）；`trim_tail()` 同时被 flush 的第三层过滤器与最终离线解码复用
- 日志措辞由 `Segmenter` 生成（`Action.flush_line`/`debug_line`），单 lane 模式下 `tag` 为空，输出与重构前逐字一致

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

## 7. 波束路由与固定波束

### 7.1 输出路由（`BeamRouter`）

`recon()` 只读探测 → `describe()` 检测映射歧义（歧义时拒绝写入）→ `apply()` 先存原路由、写后验证、失败 `restore()` 回滚。门控/推理通道与候选波束通道相互独立。

### 7.2 固定双波束（`FixedBeamController`）

设备只有 **2 个可编程固定波束**（beam1/beam2），另有 free-running 与 auto-select 两个由固件派生、不可编程。把 beam1/beam2 分别指向两个已知座位、并开启能量门控，是硬件本身对"两人串音"的答案：能量低的那一侧被**静音**而非混入，因此送进识别的音频里只有一个人。

- 写入序列：`AZIMUTH` → `ELEVATION` → `GATING` → **`SONOFF` 最后**
- `steer()` 全程存原值，失败或退出时 `restore()` 回滚

### 7.3 实测结论（firmware 2.0.8，2026-09-18）

以下均为设备实测，非文档推断：

| 结论 | 证据 |
|---|---|
| **方位单位是弧度** | 写 `1.5708` → 读回 `90.0`；若误写 `90.0` 则读回 `5156.6`（越界） |
| **指向精确跟随** | 写 (0,180)(30,210)…(330,150) 十二点，`AEC_AZIMUTH_VALUES[0..1]` 与写入值误差 < 0.5°，**12/12 命中** |
| **不需要先写仰角** | 省略 `ELEVATION` 时方位照样跟随（曾怀疑是前提，实测否定） |
| **逐波束能量有效** | beam1 指向扫过一圈时 `SPENERGY[0]` 从 107k → **378k 峰(120°)** → 归零，随指向显著变化 |
| **能量选波束机制成立** | `SPENERGY[3]`(auto-select) 恒等于 `max(SPENERGY[0], SPENERGY[1])`，12/12 样本成立 |
| **固定模式下自由波束停用** | `SONOFF=1` 时 `SPENERGY[2]`(free-running) = 0 |
| **方位读数会漂移** | `AEC_AZIMUTH_VALUES[2]/[3]`（free/auto，`DOA_VALUE` 报的就是它）空房间内 5 次读到 89→58→181→43→182，**不适合作为方位判据** |

**待验证**：`SPENERGY` 的"真人说话时两波束是否分离"尚未测过。此前的环境噪声源间歇出现，同一角度两次读数可差数倍，因此"能量有效"已证实，但"能分离两个说话人"仍属推断。

### 7.4 注意事项

- **`AEC_FIXEDBEAMSONOFF` 只能在运行期配置**：写入固件配置并保存可能导致设备重启后挂起。本项目**从不持久化**——见 `usb_device.FLASH_PERSIST_CMDS` 的守卫（4 层，任一路径都拦得住）
- **扫描前必须读回 `SONOFF` 确认生效**，并留 0.3s 以上稳定期。曾出现一次 36 点扫描方位读数被二次换算的异常（32 倍缩放），三次受控重跑均未复现，成因未明
- 扫描期间开 `SONOFF` 会改变输出音频（ch0/ch1），因此扫描应在开始字幕**之前**一次性完成

### 7.5 工具

```
--beam-probe          只读巡检全部波束参数（度数/弧度双显）
--beam-scan           扫描找峰：自动定位说话人方位
--beam-fixed AZ1 AZ2  手工指定两波束方位（度），默认扫完即回滚
--beam-fixed-no-gating  关闭能量门控
--beam-apply-in-run   运行期保持，不自动回滚
--beam-recon / --beam-select  输出通道路由探测与切换
```

## 8. 延迟预算（基线）

```
译文端到端 ~1.0s = silence_hold 0.6s(~60%) + 轮量化 ~10% + ASR 0.15~0.25s + NLLB 0.2~0.3s + WS <20ms
```

推理仅占 ~0.4s，优化重点在切句策略（软件 VAD 连续低概率帧提前 flush）而非模型。翻译（600M）增量 ~0.2–0.4s，因与 source 广播解耦，不阻塞原文显示。
