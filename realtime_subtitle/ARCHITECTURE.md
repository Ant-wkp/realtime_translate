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
入站  cmd: get_target_langs / set_target_langs / get_zones / set_zones
```

`translation` 只带 `id`——它按 id 归属，前端已在卡片上保留该 id。

设备参数命令（`list_commands`/`read`/`write`）已随前端「USB 命令」面板一并移除：本项目不再提供任何界面上的参数读写入口（见 §8）。

延迟测量只留在后端终端（`[首字]` / `[耗时]` 行 + 退出汇总），前端是纯渲染器，不接收任何 latency 字段。

## 6. 前端

单文件 `web/index.html`（后端只有 `/` 与 `/index.html` 路由，CSS/JS 全内联）。深色玻璃拟态；字幕状态机 partial 即时整段替换（后端发的是累积文本）→ final 整段替换 → discard 删除 → translation 填充；无逐字动画，每条消息到达即刻上屏。卡片上限 50 条；断线指数退避重连；XSS 全量转义。

顶部门控柱是四根 `#gate i`，由 `vad.state` 的 `hw_vad`/`segment_state` 驱动，`title` 显示门控模式。卡片 `.meta` 有 `partial` 与方位两个 pill，方位在段结束时才填。

文本一律经 `esc()` 转义：Zone 标签是**用户可编辑的自由文本**，是真实注入面（已用标记载荷实测：渲染为字面文本、不创建元素）。

设置抽屉只剩两组：翻译方向（改动即生效并持久化，见 §7.2）与 DOA 方位标签。「高级 / 设备参数」区块（USB 命令列表、读/写按钮）已移除，前端不再发送任何设备参数命令。

## 7. 延迟预算（基线）

```
译文端到端 ~1.0s = silence_hold 0.6s(~60%) + 轮量化 ~10% + ASR 0.15~0.25s + NLLB 0.2~0.3s + WS <20ms
```

推理仅占 ~0.4s，优化重点在切句策略（软件 VAD 连续低概率帧提前 flush）而非模型。翻译（600M）增量 ~0.2–0.4s，因与 source 广播解耦，不阻塞原文显示。

### 7.1 翻译侧的实测延迟（beam 与平台）

翻译是段级串行、每句一次 `translate_batch`，所以延迟直接按"每句毫秒"度量。**beam 宽度是主要杠杆**：它只乘解码次数，而 600M 模型单句解码本身就受权重加载/编码器主导。

| 配置 | x86-64（8 线程） | AGX Orin（8×A78AE） |
|---|---|---|
| `int8` beam=4 | 246 ms | 1026 ms |
| `int8` beam=1（当前默认） | **198 ms** | **828 ms** |
| `int8_float32` | 426 ms | 835 ms |
| `float32` | 995 ms | 2547 ms |

beam 1→4 只换来极小的措辞差异（`It's a nice day.` / `The weather is fine today.` / `The weather is good today.`），而代价是 +25%，所以默认取贪心；`--nllb-beam` 可调回 4。**该差异只在三个句子上做过目测，不是语料级评测**，不要把它当成"beam=1 与 4 等价"的结论。

两个反直觉的负结果：

- `intra_threads` 在 x86 上无收益（0/4/8 = 343/328/429 ms）；在 Orin 上**设成 8 反而慢一倍**（1101 → 2271 ms），因为 8 个核全被占满后 ASR 与翻译互相抢核。默认不动它。
- 把 CPU governor 从 `schedutil` 改成 `performance` **没有可测改善**，所以脚本不要求改 governor。

Orin 比 x86 慢约 4 倍，根因不在 CPU 频率而在**后端差异**：同一份 PyPI aarch64 轮子用 `GEMM_S8 backend: Ruy`，而 x86 轮子用 Intel MKL/oneDNN（`CT2_VERBOSE=1` 可见那行 "SGEMM backend / GEMM_S8 backend"）。Ruy 的 int8 kernel 明显弱于 MKL。因此 Orin 上提升翻译速度的正确方向不是调参，而是**让它用 GPU**。

按 [requirements-jetson.txt](requirements-jetson.txt) §3 编译带 CUDA 的 ctranslate2 后（`CUDA_ARCH_LIST=8.7`，Orin 原生 sm_87，无 PTX JIT），实测：

| 配置 | Orin 每句 |
|---|---|
| CPU `int8`（PyPI 轮子，Ruy） | 828 ms |
| GPU `float16`（源码编译，当前默认） | **346 ms**（ASR 并发下实测中位） |

### 7.1.1 GPU 量化类型的选择 —— 以可靠性而非速度为准

GPU 上试过三种量化类型，**`float16` 是唯一在真实运行条件下不出错的**。关键在于“真实条件”：ASR 也常驻同一块 GPU，而 int8 路径会调用 cuBLAS 的 int8 GEMM kernel，在另一路 CUDA 负载存在时会失败：

| compute_type | 与 ASR 并发时 | 中位延迟 |
|---|---|---|
| `int8_float16` | **6/24 成功** | 320 ms |
| `float16` | **24/24 成功** | 406 ms |

`float16` 用约 +27% 的延迟换来 100% 可靠性 —— 字幕整个丢失比晚 90 ms 严重得多。这个差值实测复现两次。

**排查上的教训**：单独压测（不并发）时**六种量化类型全部 15/15 通过**，因此一度误判为已修好；只有把 ASR 并发加进来才复现出 6/24。验收用例必须包含“另一路 CUDA 负载同时在跑”，否则会得到一个假阳性。

即使默认已是可靠配置，`_translate_batch()` 仍保留一层兜底：GPU 上遇到 `RuntimeError` 时用 `float32` 重试一次（懒构造，仅失败时才占显存），因为用户可用 `--nllb-compute-type int8_float16` 选回不可靠路径，且驱动层瞬时错误总是可能。CPU 侧不重试 —— 没有可降级的目标，静默重试只会掩盖真问题。

一次真实会话（20 段）里 GPU 翻译落在 216–411 ms，`cuBLAS` 失败 0 次。ASR 占约 3 GB 显存，与翻译共存无冲突。

**必须知道的陷阱**：用 `CUDA_DYNAMIC_LOADING=ON` 编译时，ct2 在运行时按 soname dlopen cuBLAS。**不要为了“帮它找到库”而用 `ctypes` 预先再加载一份** —— 实测有害：把同一 soname 的第二个副本塞进进程后，GEMM 会选到无法服务该矩阵形状的 kernel，表现为 `CUBLAS_STATUS_NOT_SUPPORTED`。在长短句混合的 40 次里：不预加载 **20/20 成功**，预加载 torch 版 **10/20 失败**。失败是**按输入文本确定性的**（某些句子必失败，同长度的另一些从不失败），而非库冲突该有的间歇性 —— 这也是区分两者的关键。`compute.ct2_cuda_library_status()` 因此只**报告** soname 可否解析、绝不加载任何东西。

### 7.2 翻译方向的持久化

`runtime_settings.json`（包目录旁）保存 `zh_to` / `other_to`。这**不是**设备侧持久化——那是被 `usb_device.py` 拦截的东西，因为写 flash 的波束开关可能让设备重启后挂起；这里只是一个普通 JSON 文件。加它的起因是一个真实误报：方向原本硬编码为 `zh_to="en"` 且不存盘，**重启后设置静默丢失**，看起来就像"选了日语仍然翻译成英语"。读取时机容错：文件损坏回退默认值，语言码不在 `LANG_TO_NLLB`/`TARGET_LANGS` 里则丢弃，绝不让一个坏文件挡住启动。

## 8. 波束：为什么不再由程序配置

设备有 2 个可编程固定波束（beam1/beam2，`AEC_FIXEDBEAM*`），另有 free-running 与 auto-select 两个由固件派生。程序**不再**在启动时转向或改路由：目标场景是"噪声里稳定识别一个人"，而固件默认输出（`AUDIO_MGR_OP_ALL` 出厂配置，L=处理输出、R=ASR 波束输出）本身就在做这件事。用软件重配会引入必须回滚的设备状态，却没换来识别质量。

因此启动流程不再写设备任何参数。波束相关的读写**再无任何界面入口**：前端「USB 命令」面板已随 `control.py` 的 `list_commands`/`read`/`write` 一并移除（理由同上——这段能力没有换来识别质量，却要带着一套必须回滚的设备状态）。

> 设备层的写入能力与 flash 守卫**也已一并移除**（见 §8.2），`usb_device.py` 现在只读。§8.1 的结论仍然有效，是将来若要重新启用波束的起点。

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

### 8.2 从不持久化：现在由架构保证

`usb_device.py` **只读**。它曾经提供 `write()` 并配三个集合拦截会写 flash 的命令（`SAVE_CONFIGURATION`/`CLEAR_CONFIGURATION`）与会重启/重置的命令（`REBOOT`/`USB_BIT_DEPTH`/`TEST_CORE_BURN`），抛 `PermissionError`。

那套守卫连同它唯一的调用方（前端「USB 命令」面板）已一并移除。**结论不变但保证方式变了**：本项目仍然从不持久化 —— `AEC_FIXEDBEAMSONOFF` 写进固件后保存可能导致重启挂起 —— 但现在不是"有写入路径但被拦住"，而是**根本没有写入路径**。这比守卫更强：守卫只覆盖它枚举到的命令，而"没有入口"覆盖全部。

代价是失去了运行期写设备参数的能力（含只读之外的调试用途）。若要重新启用，需同时恢复写入路径与守卫；§8.1 的实测结论仍然有效，是那时的起点。

### 8.3 采集通道

采集流恒为 **6 通道**，程序**固定取索引 0**（`main.py` 的 `CAPTURE_CHANNEL`，横幅写作 ch1），即"处理输出"——AEC+波束+后处理的完整链，也就是本程序依赖的那一路已完成的工作。其余索引不提供 CLI 入口，仅记录如下：

| 索引 | 横幅 | 内容 |
|---|---|---|
| 0 | ch1 | 处理输出（AEC+波束+后处理）—— **固定使用** |
| 1 | ch2 | ASR 波束输出（auto-select，无后处理） |
| 2–5 | ch3–ch6 | 四路原始麦克风 |

`--audio-source`/`--ch`/`--in-channels` 三个参数的唯一作用是选索引，已移除。选择固定的代价是失去 ch1 与四路原始麦的运行时访问；换通道改 `CAPTURE_CHANNEL` 一个常量。

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
