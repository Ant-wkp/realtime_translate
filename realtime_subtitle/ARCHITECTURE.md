# 架构

一句话：XVF3800 硬件负责音频增强与空间感知，Python 单进程多线程流水线负责门控/分段/识别/翻译，浏览器负责显示与运行时控制。

## 1. 执行流（6 + 1）

| # | 执行流 | 类型 | 节拍 | 职责 |
|---|---|---|---|---|
| 1 | audio_callback | PortAudio 线程 | 按块 | 从 6ch 采集里取出选定通道，入 audio_queue |
| 2 | VadDoaReader | daemon | 100ms | USB 读 DOA_VALUE（语音标志 + 方位），入 vad_queue |
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

模式只有一种：**每拍合取**（硬件标志 AND `sw_prob >= 0.35`）。两者都不能单独用——固件的标志是纯能量门，响度够就开（门铃也算）；Silero 是模型，持续噪声能把它骗成假阳性。合取后段级过滤还会再过一遍。

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
- 超长段强制切分；DOA 众数随段保存（仅展示，不参与过滤）
- 实现见 `segmenter.py`：`Segmenter` 持有全部状态，`feed(chunks, gate_fn, hw_vad, now)` 每次 tick 返回 `Action` 列表（`open`/`feed`/`flush`/`discard`），由 `main.py` 的 `apply_actions()` 翻译成对转录器与前端广播的调用。软 VAD 概率在 `Segmenter` 内部计算一次，同时驱动门控判定、统计与句尾确认，避免同一个 tick 出现多个读数
- 该文件是纯逻辑、无 I/O，可脱离硬件测试（`tests/test_segmenter.py`）；`trim_tail()` 同时被 flush 的第三层过滤器与最终离线解码复用
- 日志措辞由 `Segmenter` 生成（`Action.flush_line`/`debug_line`），与重构前的内联循环逐字一致

## 4. 推理层

| 执行体 | 位置 | 模型 |
|---|---|---|
| sw VAD | 主循环内联 | Silero torch jit（官方 silero-vad 包 / 本地 silero_vad.jit） |
| ASR | TranscriberWorker | FunASR paraformer-zh + ct-punc |
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
      final{id,ts,doa,text,lang}
      discard{id}
      translation{id,ts,text,target_lang}
      vad.state{hw_vad,sw_prob,gate_mode,segment_state}
      cmd_result{ok,data|error}
入站  cmd: list_commands / read / write / get_target_langs / set_target_langs / get_zones / set_zones
```

`translation` 只带 `id`——它按 id 归属，前端已在卡片上保留该 id。

延迟测量只留在后端终端（`[首字]` / `[耗时]` 行 + 退出汇总），前端是纯渲染器，不接收任何 latency 字段。

## 6. 前端

单文件 `web/index.html`（后端只有 `/` 与 `/index.html` 路由，CSS/JS 全内联）。深色玻璃拟态；字幕状态机 partial 即时整段替换（后端发的是累积文本）→ final 整段替换 → discard 删除 → translation 填充；无逐字动画，每条消息到达即刻上屏。卡片上限 50 条；断线指数退避重连；XSS 全量转义。

顶部门控柱是四根 `#gate i`，由 `vad.state` 的 `hw_vad`/`segment_state` 驱动，`title` 显示门控模式。卡片 `.meta` 有 `partial` 与方位两个 pill，方位在段结束时才填。

文本一律经 `esc()` 转义：Zone 标签是**用户可编辑的自由文本**，是真实注入面（已用标记载荷实测：渲染为字面文本、不创建元素）。

## 7. 延迟预算（基线）

```
译文端到端 ~1.0s = silence_hold 0.6s(~60%) + 轮量化 ~10% + ASR 0.15~0.25s + NLLB 0.2~0.3s + WS <20ms
```

推理仅占 ~0.4s，优化重点在切句策略（软件 VAD 连续低概率帧提前 flush）而非模型。翻译（600M）增量 ~0.2–0.4s，因与 source 广播解耦，不阻塞原文显示。

## 8. 波束：为什么不再由程序配置

设备有 2 个可编程固定波束（beam1/beam2，`AEC_FIXEDBEAM*`），另有 free-running 与 auto-select 两个由固件派生。程序**不再**在启动时转向或改路由：目标场景是"噪声里稳定识别一个人"，而固件默认输出（`AUDIO_MGR_OP_ALL` 出厂配置，L=处理输出、R=ASR 波束输出）本身就在做这件事。用软件重配会引入必须回滚的设备状态，却没换来识别质量。

因此启动流程不再写设备任何参数；波束相关的读写只在**前端「USB 命令」面板**里按需手动执行（只读命令随时可读；会进 flash 的命令被 `usb_device.py` 的守卫拦住，见 §8.2）。

### 8.1 实测结论（firmware 2.0.8）

以下均为设备实测，非文档推断（这些数字仍成立，是将来若要重新用波束的起点）：

| 结论 | 证据 |
|---|---|
| **方位单位是弧度** | 写 `1.5708` → 读回 `90.0`；若误写 `90.0` 则读回 `5156.6`（越界） |
| **指向精确跟随** | 写 (0,180)(30,210)…(330,150) 十二点，`AEC_AZIMUTH_VALUES[0..1]` 与写入值误差 < 0.5°，**12/12 命中** |
| **不需要先写仰角** | 省略 `ELEVATION` 时方位照样跟随（曾怀疑是前提，实测否定） |
| **逐波束能量有效** | beam1 指向扫过一圈时 `SPENERGY[0]` 从 107k → **378k 峰(120°)** → 归零，随指向显著变化 |
| **能量选波束机制成立** | `SPENERGY[3]`(auto-select) 恒等于 `max(SPENERGY[0], SPENERGY[1])`，12/12 样本成立 |
| **固定模式下自由波束停用** | `SONOFF=1` 时 `SPENERGY[2]`(free-running) = 0 |
| **静音时 `SPENERGY` 恒为 0** | 空房间 0～42916，拍手峰值 2.4M～4.3M（差 50 倍以上）。**交叉验证必须在发声期间轮询**：拍手之前读只会得到全 0 |
| **方位读数会漂移** | `AEC_AZIMUTH_VALUES[2]/[3]`（free/auto，`DOA_VALUE` 报的就是它）空房间内 5 次读到 89→58→181→43→182，**不适合作为方位判据** |

**未验证**：`SPENERGY` 能否在两人同时说话时分离两个波束——这正是放弃双波束的间接原因之一：6.7cm 孔径的物理上限没人测过，而单波束方案不依赖它。

### 8.2 flash 守卫

`usb_device.py` 把命令分三类，只有第三类需要额外授权：

| 集合 | 含义 | 行为 |
|---|---|---|
| `FLASH_PERSIST_CMDS` | 会写 flash（`SAVE_CONFIGURATION`/`CLEAR_CONFIGURATION`） | `write` 抛 `PermissionError`，除非显式 `allow_persist=True` |
| `DISRUPTIVE_CMDS` | 会重启或改 USB 链路（`REBOOT`/`USB_BIT_DEPTH`/`TEST_CORE_BURN`） | 同上 |
| `GUARDED_CMDS` | 以上两者的并集 | 前端命令面板标记 `blocked` + `block_reason` |

本项目**从不持久化**：`AEC_FIXEDBEAMSONOFF` 写进固件后保存可能导致重启挂起，所以任何路径都拦得住。

### 8.3 采集通道

采集流恒为 **6 通道**，但不提供 `mic0`–`mic3` 与 `--ch` 之外的入口：`--audio-source` 只有两项，其余靠 `--ch N` 直指。

| 通道 | `--audio-source` | 内容 |
|---|---|---|
| ch0 | `conference`（默认） | 处理输出（AEC+波束+后处理） |
| ch1 | `asr` | ASR 波束输出（auto-select，无后处理） |
| ch2–ch5 | ——（用 `--ch 2..5`） | 四路原始麦克风 |

命名的是**固件 source 槽位**，不是物理方向：方形阵列旋转对称，循环移位不可观测，故只能确定"环形序"、无法区分正/反向的 8 种等价解。这对字幕用途无影响（选任一颗麦克风都是确定的一颗）。

`AUDIO_MGR_OP_ALL` 的 12 个字节是 6 组 `<category>, <source>`。**category 号是观测值，不是文档枚举**（XMOS 公布了 pair 格式，未公布号码）：

| category | 含义 | 依据 |
|---|---|---|
| 8 | 处理后的 "Conference" 输出 | 设备出厂 `L = (8,0)`，与 ch0 描述吻合 |
| 7 | ASR 输出路径，**source = 波束索引** | 实测：`(7,0)`→ch0、`(7,1)`→ch1 |
| 1 | 逐麦克风信号 | 设备在四个次级槽填了 `(1,0)(1,2)(1,1)(1,3)`，source 0–3 恰是四路 |
| 4 / 12 | 远端参考：原始 / 含增益与延迟 | 出自 XVF3800 文档（唯一有具体数值的映射） |

`OP_L`/`OP_R` 只决定**前两个**通道装什么，通道 2–5 是固定的四路原始麦克风，不受路由影响。判据方法（远端参考注入）：把两槽切到 `(4,0)`/`(12,0)`，远端参考在无播放时是数字静音（精确 0.0），切后 ch0/ch1 跌至 −158/−189 dB 即精确零，而 ch2–5 仅动 +0.5～0.7 dB。用静音而非播放音频，是因为一播声 ch2–5 作为麦克风也会响，就分不清"路由改变"与"声学耦合"。

外部资料称 ch2–5 的 category 为 3 而本机实测为 1，且称其"不可更改"而路由表次级槽有值——两者不冲突：`AUDIO_MGR_OP_PACKED=[0,0]` 时只有每侧首源进 L/R，PK1/PK2 里的值是惰性默认。

### 8.4 方法上的坑（勿重复踩）

- **"复盘式"对比（两次独立录音）不成立**：它假设说话人站着不动。实测三次独立 session 中，两个**完全相同**的配置给出矛盾结果，电平还掉了 10 倍。改为**一次连续录音内切换**后，漂移变成慢变且两通道共享，在通道**平衡值**里自动抵消，而配置切换是阶跃
- **单次翻转不构成证明**：两通道下无效应时也有 50% 概率翻转。必须要求**每次切换都跟随**（7 次全中，偶然概率 1/64）
- **判据用效应量，不用相关系数、也不用 dB 阈值**：小而**完美重复**的 ±1 dB 是真实证据（噪声不会规律同相），大但有组内散布的摆动则不是。本项目用 `组间均值差 vs 组内散布`
- **拍手是瞬态**，用整窗 RMS 会把它稀释约 30 dB。应取**短窗峰值**（25ms 窗 RMS 的最大值）：合成数据下区分度由 8 倍提升到 34 倍
