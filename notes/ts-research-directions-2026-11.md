# 时序论文库梳理与可继续挖掘的方向（截至 2026-11-19）

> **范围**：本仓库 `paperlib` 中的全部文件，共 11 个 PDF。其中 10 篇与时序相关，1 篇是 Transformer 原文，另有 1 个 Visio 画图模板（不是论文）。每篇都读了方法、实验和局限部分。
> 另外做了十余次联网检索，用来判断 2025–2026 年哪些方向已经拥挤。检索只看到摘要，没有逐篇细读，文中涉及外部工作时会注明。
> 文中标“**我的判断**”的地方是推断，不是论文原文结论。

---

## 0. 结论先行

1. **库里的论文是一条主线加两条支线**
   - **主线：多变量预测（MTSF）中的跨变量依赖建模。** 依次是 CACRN（聚类）、DyTimeNet（训练期随机稀疏）、MPGTimer（相似度图偏置注意力 + 长上下文）、MoVE（时序专家 × 跨变量专家）、GPS-Mamba（图置换扫描 SSM）。
   - **支线 A：用 LLM / 基础模型做预测。** 从 Time-LLM 开始，接着是《Are LMs actually useful》的消融否定，再到 SCALER（LLM 只负责粗到细的测试时细化）。另有一篇基础模型综述。
   - **支线 B：Mamba / SSM。** 从 S-Mamba 到 GPS-Mamba。
2. **精读后最扎实的两个空白**
   - **空白一：五篇跨变量论文都说自己建模了“动态/稀疏”依赖，但没有一篇在推理期真正实现了“显式、按样本自适应、稀疏、变量级”的依赖结构。** 逐篇拆解见 §3.1。
   - **空白二：它们都观察到同一现象，但都没有解释。** 强耦合数据（PEMS、SD、Solar、Weather）上有收益，弱耦合数据（ECL、ETT）上收益很小，甚至变差。没有人给出一个能事先判断“跨变量建模有没有用”的标准。
3. **推荐优先挖的方向**（详见 §5）
   - ① **预测窗口感知、按周期相位条件化的跨变量依赖。** 思路是把 CycleNet/TQNet 的周期性从“数值”推广到“关系”，并支持多周期。新颖度高，工作量小，单卡可做。
   - ② **“公共因子 + 稀疏残差”的依赖建模，加上“依赖可用性”诊断。** 既能解释为什么 CI 常常够用，也能说明什么时候该用 CD。
   - ③ **SSM 的变量扫描顺序。** 先解释 S-Mamba 和 GPS-Mamba 结论相反的原因，再做可微或层级排序。
   - ④ **自适应测试时计算。** 先对 SCALER 做 Tan 等人那种受控消融，再做“按样本难度分配算力”的轻量粗到细预测。
4. **不建议再投入的方向**
   - 把 LLM 当数值预测骨干的 reprogramming 类工作。
   - 在 ETT/Weather/ECL 上用单个随机种子刷 0.00x 的 SOTA。
   - 单纯把 Attention 换成 Mamba。
   - 单纯做测试时适应（TTA），2025–26 年已经很拥挤。

---

## 1. 库内论文一览

| 文件编号 | 论文 | 出处 | 类别 | 核心做法 | 实验要点 |
|---|---|---|---|---|---|
| 4 | Attention Is All You Need | NeurIPS 2017 | 基础架构 | 自注意力 Transformer | — |
| 1 | Foundation Models for Time Series Analysis: A Tutorial and Survey | KDD 2024 | 综述 | 从架构、预训练、适配、数据模态四个角度给时序基础模型（TSFM）分类 | 列出的未来方向：多模态、高效架构（如 Mamba）、分布漂移与因果、隐私与联邦 |
| 2 | Time-LLM | ICLR 2024 | LLM | 冻结 LLaMA-7B；把 patch 重编程到文本原型上，再用 Prompt-as-Prefix 补充上下文 | 覆盖长期、短期、少样本、零样本 |
| 3 | Are Language Models Actually Useful for TSF? | NeurIPS 2024 | LLM 批判 | 对 OneFitsAll、Time-LLM、CALF 做三种消融：去掉 LLM、换成注意力、换成 Transformer | 13 个数据集。去掉 LLM 或随机初始化都不掉点，算力却贵 1–3 个数量级；打乱输入顺序不影响结果；少样本也没有帮助 |
| 5 | SCALER: Efficient Test-Time Scaling for LLM-based TSF | KDD 2026 | LLM + 测试时扩展 | 轻量 Transformer 先预测下采样的“粗形状”，再由 LLaMA-7B 按固定 K 步做粗到细的残差细化 | L=672，逐通道（CI）处理；比 TimeReasoner 快约 6.7–6.8 倍 |
| 6 | Is Mamba Effective for TSF?（S-Mamba） | Neurocomputing 2025（库中为 arXiv 2024 版） | SSM | 每个变量作为一个 token；变量间相关用双向 Mamba 建模，时间依赖用 FFN 建模 | 13 个数据集，L=96；还测了变量顺序、长 lookback、未见变量泛化 |
| 7 | CACRN | Neurocomputing 2023 | 跨变量（聚类） | 先对变量做模糊 C 均值聚类；簇内用 FM 式两两交互；每个簇接一个 CNN+BiLSTM；最后做簇内和簇间注意力 | 短期预测（h=3–12），窗口 12–36，指标 RMSE/MAE；基线较旧；代码需向作者索取 |
| 8 | DyTimeNet | Neurocomputing 2026 | 跨变量（稀疏） | 倒置 MLP 做通道混合，配合 DropFusion（DropConnect + Dropout）；时间维用 FFN | 13 个数据集，L=96；参数约为 iTransformer 的 1/4，训练快约 7 倍 |
| 9 | MPGTimer | Neurocomputing 2026 | 跨变量 + 长上下文 | 基于 Timer-XL；多尺度 patch 融合，加图增强注意力 S = QKᵀ + M + G | L=672，7 个数据集；MSE 比 Timer-XL 低 3.13%，但在 ECL 上不如 Timer-XL；没有测 Traffic |
| 10 | MoVE | ICLR 2026 投稿（匿名） | 跨变量（专家混合） | 用 CycleNet 的周期向量，加跨变量编码器 RCVTE，共 4 条专家通路，由门控加权；另有 PALCL 课程学习和 RCVA 注意力 | L=96，单一随机种子；16 项指标中 7 项最优，Traffic 和 Weather 不是最优 |
| 10 | GPS-Mamba | ESWA 2026 | 跨变量 + SSM | 用 Laplace 扫描（谱排序）确定变量顺序；静态顺序负责长期相关，从仓库检索的动态顺序负责短期变化 | 7–8 个数据集（含 SD、PEMS、NN5），L=96；80 项中 56 项第一 |
| — | Drawing template.vsdx | — | 非论文 | Visio 画图模板 | — |

---

## 2. 研究脉络

```
Transformer (2017)
 ├─ 时序 Transformer：Informer / Autoformer / FEDformer（每个时间点一个 token，隐式混合通道）
 ├─ CI 派：DLinear / PatchTST / CycleNet / SparseTSF（每个通道独立建模，稳健）
 └─ CD 派：Crossformer / iTransformer / TimeXer / TQNet（显式建模跨变量）
        ├─ 聚类/分组 ……… CACRN(2023)            同类后续：CCM、DUET(KDD'25)
        ├─ 稀疏化 ………… DyTimeNet(2026)         同类：TimeFilter(ICML'25)、MS-FLOW(2026)、Li-Net(ICDE'26)
        ├─ 图偏置注意力 + 长上下文 … MPGTimer(2026, 基于 Timer-XL)
        ├─ 专家混合 ……… MoVE(2026 投稿, 基于 CycleNet/TQNet)    同类：PatchMoE、Dynamic TMoE
        └─ SSM 扫描 ……… S-Mamba(2024) → GPS-Mamba(2026)      同类：MambaTS、SOR-Mamba

LLM / 基础模型线：
  Time-LLM(2024)，被 Tan et al.(2024) 质疑（LLM 可以去掉）
      └→ SCALER(2026)：LLM 只做粗到细细化，强调测试时计算的效率
  综述(KDD'24) → 2025–26：Chronos-2 / TimesFM 2.5 / Moirai 2.0 / Timer-XL / Sundial / Timer-S1 ……
```

---

## 3. 横向分析：共同的隐含假设与短板

### 3.1 “动态/稀疏依赖”大多名不副实（最重要）

下表按各论文自己的公式和描述，逐篇拆解它们宣称的机制实际是什么。

| 论文 | 宣称 | 实际机制 | 推理期能否按样本自适应？是否显式稀疏？ |
|---|---|---|---|
| CACRN | 分离无关变量 | 模糊 C 均值在主网络之外单独求解，簇数 k 仍要在验证集上搜索 | 基本是静态的；软隶属，不稀疏 |
| DyTimeNet | 动态稀疏依赖 | DropFusion 只在训练时随机丢边、丢神经元；测试时去掉掩码，按比例缩放 | 通道混合权重在推理期固定，只靠激活函数带来很弱的输入依赖；推理期不稀疏 |
| MPGTimer | 图增强的动态依赖 | 图项 G_ij = (HW_g)_i · (HW_g)_j，和 QKᵀ 一样是双线性相似度。两者相加等于给注意力打分加一个对称的低秩项，并没有引入显式的变量级图；结构掩码 A 默认全连接 | 自适应程度和普通注意力相同；不稀疏，复杂度 O((C·N)²) |
| MoVE | 稳健的跨变量注意力 + 动态门控 | RCVA = Softmax(GELU(A − LSE(A)))，见下方推导 | 注意力接近均匀，约等于跨通道取平均；门控值也接近均匀 |
| GPS-Mamba | 动态扫描顺序 | 动态顺序来自预先构建的有限仓库（默认 50 个排列），用冻结的 S-Mamba 嵌入做最近邻检索；仓库里的排列由训练样本**未来窗口**的相关性算出 | 只能在有限候选中离散挑选，不可微；训练期存在轻微“看未来”，训练和测试不一致（作者自己承认） |

**MoVE 的 RCVA 推导（我的判断）**

这是按论文式 (3) 推出来的。如果官方代码和公式不一致，以代码为准。

- A − LSE(A) 就是 log-softmax，取值都 ≤ 0。
- GELU 在 (−∞, 0] 上的取值只落在 [−0.17, 0] 之间。
- 再过一次 softmax，任意两个变量的权重之比都不超过 e^0.17 ≈ 1.19。
- 变量一多（例如 ECL 有 321 个），log-softmax 大多在 −log C ≈ −5.8 附近，GELU 输出几乎都是 0，注意力基本就是**均匀平均**。
- 这个映射还是**非单调**的：最相关和最不相关的变量都会拿到接近最大的权重。

所以它“防止注意力塌缩”的方式，其实是取消了选择性。另外，论文附录 Table 6 中的门控值：周期输入下 Expert1 = 0.320，其余专家在 0.205–0.241 之间，整体接近均匀。这和正文所说的“0.68 / 0.72”对不上。

**结论**：在库内论文里，真正做到“推理期、按样本、显式稀疏、变量级、可验证”的依赖结构仍是空白。不过外部已经有人开始做：TimeFilter（ICML 2025）、DUET（KDD 2025）、MS-FLOW（2026 年 5 月预印本）、Li-Net（ICDE 2026）。所以要做这个方向，必须有更明确的差异化，见 §5 的方向 ① 和 ②。

### 3.2 “跨变量建模什么时候有用”没有被回答

几篇论文的观察：

- **MPGTimer**：在 ECL 上输给 Timer-XL。作者的解释是，ECL 的变量相关主要来自“时间对齐的公共周期”，而不是结构性的交互。
- **GPS-Mamba**：在 Weather、ECL 上增益很小，在 SD、PEMS 上增益很大。
- **DyTimeNet 的消融**：Weather 更依赖跨变量模块，ETT/ECL/Traffic/ILI 更依赖时间模块。而且把跨变量模块换成自注意力，比直接删掉还差。
- **MoVE 的消融**：在 Solar 和 ETTm2 上，同时去掉 RCVA 和门控，结果在部分指标上反而更好。

每个模型都靠一个“旋钮”控制要用多少跨变量信息，而且都只能逐个数据集调：

| 模型 | 旋钮 |
|---|---|
| CACRN | 簇数 k |
| DyTimeNet | 丢弃率 p1/p2，最优值从 0.3 到 0.8 不等 |
| GPS-Mamba | 稀疏阈值 ε 和仓库大小 n_s，n_s 最优值从 25 到 200 不等 |
| MoVE | 门控 |

**缺的是两样东西**：一个能在训练前算出来、并能预测“CD 相对 CI 有多少增益”的诊断量；以及与之配套的自适应机制。

### 3.3 稠密注意力不是跨变量混合的好归纳偏置

多篇论文互相印证：

- S-Mamba：在变量维上，Mamba 优于注意力。
- DyTimeNet：把跨变量模块换成注意力，比删掉它还差。
- MoVE：原始注意力会塌缩，而它的“修复”实际上是近乎均匀的混合。
- Tan et al.：一层注意力加上 patch 就能追平 LLM。

→ **我的判断**：标准基准中的跨变量信息，很可能以**低秩的公共成分**为主（比如共同的日周期），而这部分 CI 模型从各自历史里就能学到。只有少数数据集存在**稀疏的局部结构交互**（如空间邻接）。这就是 §5 方向 ② 的动机。

### 3.4 评测协议薄弱，基准趋于饱和

- **基线数字多是照抄的。** MoVE 抄 TQNet，SCALER 抄 LVICL，MPGTimer 的主表抄 Timer-XL 和原论文。lookback、超参、随机种子都不统一。MoVE 只用了一个种子（2024）。所有论文都没有做显著性检验。
- **lookback 设定不一。** MoVE、DyTimeNet、GPS-Mamba、S-Mamba 用 L=96；SCALER、MPGTimer 用 L=672。
- **改进幅度很小。** 在 ETT/Weather/ECL 上的提升常在 0.001–0.01 量级，可能还在种子方差之内。TFB 还指出了测试时 `drop_last` 之类的实现陷阱。

### 3.5 任务设定单一

全部论文都是规则采样、无缺失、点预测（只报 MSE/MAE）、不带外部文本。只有 S-Mamba 和 GPS-Mamba 测过“未见变量”的泛化。

### 3.6 LLM 线的核心矛盾仍未解决

- Tan et al. 证明：在这类数值预测里，LLM 可以删掉或随机初始化，效果不变。
- SCALER 显示 LLM 越大效果越好，但**没有做“同结构随机初始化”或“去掉 LLM”的对照**。

**我的判断**：SCALER 的增益可能主要来自“粗到细多尺度监督”这个结构本身，而不是 LLM 的预训练知识。

### 3.7 论文之间的矛盾（可以直接变成研究问题）

| 问题 | 一方 | 另一方 |
|---|---|---|
| 变量顺序重要吗？ | S-Mamba：几乎没影响（只在 Electricity 上挪动了非周期变量的位置） | GPS-Mamba、MambaTS：影响显著，尤其是高维空间数据 |
| 长 lookback 有用吗？ | Zeng et al.、S-Mamba：Transformer 类模型不随 L 变长而变好 | MPGTimer、DyTimeNet：到 672–720 仍在提升，960 以后收益递减 |
| LLM 有用吗？ | Time-LLM：有用 | Tan et al.：没用；SCALER：大模型更好，但缺对照实验 |

---

## 4. 外部进展速览（2025–2026，用来判断拥挤度）

> 来自联网检索，只看了摘要，没有逐篇精读。

| 方向 | 代表工作 | 拥挤度 |
|---|---|---|
| 动态/稀疏跨变量依赖 | TimeFilter (ICML'25)、DUET (KDD'25)、MS-FLOW（2026-05，样本自适应的稀疏信息流瓶颈）、Li-Net (ICDE'26, top-K 稀疏注意力)、PatchMoE、Dynamic TMoE | **中高** |
| 置换不变 / 未见通道 | CPiRi (ICLR'26)：冻结的 Sundial 做时间编码，加轻量空间模块，训练时打乱通道顺序 | 中 |
| TSFM 的多变量 / 协变量能力 | Chronos-2（group attention，显存随变量数线性增长）；TimesFM 2.5（协变量靠推理时外挂线性回归）；Moirai 2.0（单变量）；ICML'26 Channel Adapter（给 CI 预训练的 TSFM 注入稀疏跨通道信息） | 中高，大厂主导 |
| 变量扫描顺序 | MambaTS（ATSP + 模拟退火，配合变量置换训练）、SOR-Mamba、GPS-Mamba | 中低 |
| 测试时扩展 / 推理 | TimeReasoner、Time-R1（强化学习）、SCALER；ICLR'26 模型组合论文（用集成/选择替代测试时微调） | 中 |
| 测试时适应（TTA） | COSA (ICLR'26)、δ-Adapter (ICLR'26)、PETSA、AdaNODEs | **高** |
| 文本/上下文辅助预测 | CiK、TimesX、What If TSF、Dr-CiK、PostTime；LLM 推断因果图：Augur (ACL'26) | 中高，门槛高 |
| 长上下文 | Timer-XL、Timer-S1（8.3B MoE，上下文 11.5K）、StreamTimer (IJCAI'26) | 高，算力门槛高 |
| 周期 × 跨变量 | TQNet (ICML'25)、CycleNet；**没检索到多周期或“按周期条件化依赖”的直接后续** | **低** |
| 评测反思 | TFB、“No champions” (TMLR'26)、It's TIME (2026)、关于 lookback 的研究 | 中 |
| 不规则 / 缺失 | ASTGI、ReIMTS、marginalization-consistent flows（均为 ICLR'26） | 中 |
| 概率预测 × 跨变量 | EnTransformer (2026) 等 | 中低 |

---

## 5. 可以继续挖掘的方向（按推荐度排序）

每个方向按同样的结构写：库内证据 → 核心想法 → 和已有工作的区别 → 1–2 周能做完的验证 → 风险 → 评级。

### ① 预测窗口感知、按周期相位条件化的跨变量依赖 ★★★（首推）

**库内证据**

- DyTimeNet 图 1 和 GPS-Mamba 图 1(b) 都显示：交通数据早、中、晚的相关矩阵明显不同，而**不同日子的同一时段**模式相似。
- GPS-Mamba 的动态顺序是用训练样本**未来窗口**的相关性算出来的。作者的直觉是：如果输入还不在高峰期，而预测窗口进入了高峰期，就应该提前用高峰期的相关结构。这等于承认**真正有用的是预测窗口的依赖，而不是回看窗口的依赖**。但它的实现是离散检索，而且训练期有信息泄漏。
- CycleNet、TQNet、MoVE 都用可学习的周期向量建模“数值的周期性”，却没有建模“关系的周期性”。MoVE 自己承认只能处理单一周期，周期弱时就会失效。
- DyTimeNet、MPGTimer、CACRN 都只从回看窗口推断依赖。

**核心想法**

对每个预测时刻（或预测块）条件化依赖矩阵：

```
A(t+h) = Σ_k α_k( φ_day(t+h), φ_week(t+h), x_lookback ) · B_k  +  S(x_lookback)
```

- **B_k**：少量可学习的、低秩的“周期依赖基”。它们由**未来时刻**的日相位、周相位来索引——时间戳在预测时就是已知的。这相当于把 CycleNet 的循环队列从“值”推广到“关系”，天然支持多周期。
- **S(·)**：由回看内容生成的稀疏残差，用 top-k 或 hard-concrete 门实现，负责处理非周期事件。
- **退化机制**：周期弱时，α 会趋于常数，模型自动退化为纯内容驱动。
- **用法**：可以做成插件，接到 TQNet、iTransformer、S-Mamba 上，或者用来生成 GPS-Mamba 的扫描顺序，替代“从回看窗口推断依赖”。

**和已有工作的区别**

- TQNet：用周期 query，但那仍是“数值模板 + 一次注意力”。
- Sumba（NeurIPS 2024，GPS-Mamba 引用过）：用结构矩阵基建模动态相关，是**最接近的工作**，必须细读对比。但从 GPS-Mamba 的描述看，它没有以“预测时段的相位”为条件。
- GPS-Mamba：离散检索，仓库用未来数据构建。
- 检索中没找到“多周期 + 按预测时段条件化依赖”的直接工作。动手前建议再查一遍 KDD、ICLR 2026。

**1–2 周能做的验证**

1. **统计验证**：在 Traffic、PEMS、ECL、Solar 上，对每个样本算三个相关矩阵：回看窗口相关 R_back、预测窗口相关 R_fut、同相位的历史平均相关 R_phase。然后比较 ‖R_fut − R_back‖ 和 ‖R_fut − R_phase‖。如果后者明显更小，动机就成立。
2. **Oracle 上界**：在 iTransformer、S-Mamba 等 CD 模型里，把真值 R_fut 当作注意力偏置、掩码或扫描顺序（只用于测上界，不是方法本身），看误差能降多少。再把 R_fut 换成 R_phase，看能追回多少。如果连 oracle 都没收益，就及时止损。

**风险**：在 ETT/Weather 这类弱耦合、弱周期的数据上增益有限。需要用强周期、强空间耦合的数据讲故事，例如 Traffic、PEMS03/04/07/08、SD、ECL、Solar。

**评级**：新颖性高｜可行性高（单卡，在开源代码上改）｜拥挤度低。

### ② “公共因子 + 稀疏残差”的依赖建模，加“依赖可用性”诊断 ★★★

**库内证据**

- 按公式推导，MoVE 的 RCVA 几乎是均匀注意力，相当于把所有通道取平均再混回去，也就是一个**秩为 1 的公共因子**。可它仍然带来了增益。
- DyTimeNet 跨变量 MLP 的隐藏层宽度 C′ < C，是一个**低秩瓶颈**，再加随机稀疏；而且用稠密注意力反而不如完全不做跨变量。
- MPGTimer 自己承认，ECL 的相关主要来自公共周期。
- 所有论文都是强耦合数据增益大、弱耦合数据增益小（见 §3.2）。

→ **合理假设**：基准数据中的跨变量信息 = 少量**公共因子**（共同的日周期和趋势，CI 模型从各自历史里其实已经能学到）+ 少量**稀疏局部交互**（空间邻接、物理耦合，只有 CD 能用上）。这正是计量经济学里动态因子模型（DFM）的结构。

**核心想法**

- **模块设计**，由三部分组成：
  - 一条低秩公共因子通路（少量全局 token 或一个中心化表示）；
  - 一张按样本自适应的稀疏残差图（hard-concrete L0 门或 top-k，带“通信预算”）；
  - 一个逐通道的 CI 回退门。
- **顺手改造 DyTimeNet 的 DropFusion**：把它换成可学习的 L0 边门。这样“训练时随机丢边”就变成了“推理期保留下来的稀疏图”，丢弃率也不用再逐数据集网格搜索。
- **诊断量**：定义几个能在训练集上算出来的指标，检验它们能否预测“CD 相对 CI 的增益”，再用它们自动设定预算或门控初值。候选指标：
  - 公共因子能解释的方差比例；
  - 去掉公共因子之后，残差之间的条件预测增益（Granger 式或偏相关的稀疏程度）。
- **合成真值**：用“因子模型 + 稀疏、可切换的 VAR 残差”生成数据，检验模型学到的图能不能恢复真实结构。现有论文**全都缺这一步**，DyTimeNet 只是把学到的权重和 Pearson 矩阵画在一起对比。

**和已有工作的区别**

- SOFTS（NeurIPS 2024，STAR 模块）只有“公共核心”这一半。
- MS-FLOW、TimeFilter 只有“稀疏”这一半。
- Han et al.（CI/CD 的容量-鲁棒性权衡，库内论文引用过）和 2026 年关于“channel dependence / lookback / 数据集简单性”的研究，只描述了现象，没有给出可操作的诊断。
- 把两半结合，再加诊断和合成真值验证，检索范围内没看到直接对应的工作（需再确认）。

**两种成文方式**

- 方法论文：提出新模块。
- 分析论文：回答“跨变量依赖到底贡献了什么”，类似 Tan et al. 那样的 reality check，适合投 NeurIPS Datasets & Benchmarks 或 TMLR。

**风险**：MS-FLOW 等稀疏方法正在涌现，需要在“公共因子 + 诊断 + 可验证”这几点上拉开差距。

**评级**：新颖性中高｜可行性高｜拥挤度中。

### ③ SSM 的变量扫描顺序：解释矛盾，做可微或层级排序 ★★

**库内证据**

- **结论相反**：S-Mamba 说顺序几乎没影响，但它只在 Electricity 上测过，而且只挪动了非周期变量的位置。GPS-Mamba 和 MambaTS 说影响显著，尤其在 SD、PEMS 这类高维空间数据上。
- **GPS-Mamba 方法本身的局限**：
  - Laplace 扫描只用 Fiedler 向量做一维谱排序，多簇、多中心的结构会被打散；
  - 动态顺序只能从约 50 个预存排列里做最近邻检索，不可微；
  - 依赖 t-SNE + DBSCAN 预处理和预训练好的 S-Mamba，带来额外的 CPU 和 I/O 开销（作者承认）；
  - 把 TQNet 学到的相关矩阵换进去，也没有提升。

**核心想法**

- **先回答“什么时候顺序重要”**：用排列后邻接矩阵的带宽，或线性排列代价，来量化“扫描顺序和依赖结构的匹配程度”。然后系统地改变几个因素，测敏感度：变量数、依赖的局部性（空间稀疏 vs 全局低秩）、SSM 状态维 N、单向还是双向。
  - **我的预期**：依赖越局部、越稀疏，变量越多，状态维越小，顺序就越重要。这恰好能把两边的结论统一起来。
- **方法上的三个选项**：
  - **层级扫描**：先软聚类（借鉴 CACRN、CCM 的思路），簇内按谱序扫描，簇间用少量全局 token 交互；
  - **可微排序**：用 Gumbel-Sinkhorn 或 SoftSort 端到端地学排列；或者用多条排列（第 2、3 个特征向量、层次聚类的叶序）并行扫描后再融合；
  - **推理期在线排序**：直接由当前输入窗口（或方向 ① 的相位条件化依赖）计算排列，去掉检索仓库。

**和已有工作的区别**

- MambaTS：在推理期用 ATSP + 模拟退火定出**一个**全局顺序。
- SOR-Mamba：追求对顺序不敏感。
- CPiRi：走置换不变的路线。
- GPS-Mamba：谱排序 + 检索。
- “解释顺序敏感的条件 + 可微/层级扫描”仍有空间。

**风险**：Mamba 在时序预测领域的热度在下降；增益可能集中在交通类数据上，需要很强的实验和效率优势。

**评级**：新颖性中｜可行性中高｜拥挤度中低。

### ④ 自适应测试时计算：先证伪 LLM，再做轻量化 ★★

**库内证据**

- Tan et al. 提供了一套方法论：去掉 LLM，或换成注意力、随机初始化，做对照。
- SCALER 的三点不足：
  - 没有做同结构随机初始化或去掉 LLM 的对照；
  - 细化步数是固定的（作者在 future work 里提到了 adaptive refinement policies）；
  - 逐通道处理，完全没用跨变量信息。

**核心想法**

1. **受控检验**：在 SCALER 开源代码上，把 refiner 换成以下三种之一，看效果：
   - (a) 同结构但随机初始化；
   - (b) 2–4 层的小 Transformer；
   - (c) 去掉 refiner，只保留粗形状再上采样。

   如果 (a)(b) 不掉点，说明关键在“粗到细的多尺度监督”。SCALER 自己显示 GPT-2 版本也有竞争力，所以可以先在 GPT-2 上做，省显存。
2. **轻量 + 自适应**：用小模型做粗到细的残差细化，按样本决定细化几步。停止依据可以是预测不确定性、残差能量，或学出来的停止概率（类似 ACT / PonderNet）。困难样本（突变、峰谷）多算几步，平稳样本少算。最后画出“精度–算力” Pareto 曲线。
3. **跨变量随尺度变化**：在粗尺度（低频趋势）上做跨变量，因为那里的耦合更稳定、噪声更小；细尺度保持 CI。

**和已有工作的区别**

- TimeReasoner、Time-R1 偏重 LLM 推理和强化学习。
- SCALER 用固定步数。
- ICLR 2026 的模型组合论文研究的是“用集成或选择替代测试时微调”。
- 检索中没找到“按样本自适应分配测试时计算”的时序预测工作（需再确认）。

**风险**：如果对照实验表明 LLM 确实必要，方向就要改成“怎样便宜地用 LLM”。另外，自适应停止的训练稳定性需要专门处理。

**评级**：新颖性中高｜可行性中（对照实验要跑 7B 模型，显存需求较大）｜拥挤度中。

### ⑤ 把跨变量模块做成时序基础模型（TSFM）的适配器 ★★（加分项 / 放大器）

- **动机**：综述里预言的趋势已经成真。
  - Chronos-2 原生支持多变量和协变量；
  - TimesFM 2.5 处理协变量要靠外挂线性回归；
  - Moirai 2.0 不支持多变量协变量；
  - CPiRi 和 ICML'26 的 Channel Adapter 都在走“冻结 TSFM + 轻量跨通道模块”这条路。
- **做法**：把方向 ①②③ 的跨变量模块做成“冻结 TSFM + 轻量跨变量适配器”，在零样本和少样本多变量预测上验证。
- **价值**：实验更贴近 2026 年审稿人的口味，也能回答“到了 TSFM 时代，跨变量模块还有多大用”。
- **风险**：这个位置已被 Channel Adapter 和 CPiRi 占住，需要靠 ①② 独特的归纳偏置来区分。

### ⑥ LLM 语义先验 → 稀疏依赖图 ★（可选，适合想做 LLM + 时序的同学）

- **来源**：
  - MPGTimer 的 future work：用 LLM 解析文本事件、日志、领域知识，转换成结构约束或先验偏置；
  - Tan et al. 的建议：LLM 应该用在真正需要语言推理的地方。
- **做法**：
  - 训练前，一次性让 LLM 根据变量名、单位、传感器描述、领域文档，给出候选关系（方向、时滞、正负号），作为稀疏图的先验或正则；
  - 再用数据驱动的方式做后验校正，并学一个“先验可信度”；
  - 推理期完全不调用 LLM，从而避开 Tan et al. 的批评。
- **已有工作**：Augur（ACL'26）用 LLM 推断协变量之间的有向因果图，再用于预测。需要说清楚差异，比如：用语义元数据而不是数值片段、先验可信度自适应、显式建模时滞关系。
- **风险**：ECL、Traffic 等基准里的变量只有编号、没有语义名，需要换成 Weather、Air Quality 或工业、医疗这类带语义的数据集；先验质量也难以评估。

### 其他延伸（优先级较低，或已较拥挤）

- **概率预测 × 跨变量依赖**：库内论文全是点预测；外部已有 EnTransformer 等。
- **缺失 / 不规则采样下的依赖建模**：ICLR'26 已有 ASTGI、ReIMTS 等。
- **变量增减与未见变量泛化**：CPiRi 已经做了；S-Mamba 和 GPS-Mamba 只做过“用 40% 变量训练”这样的简单实验。
- **隐私 / 联邦训练的 TSFM**：综述提到过，竞争较少，但需要合适的应用场景。
- **长上下文**：有 Timer-S1、StreamTimer 等，算力门槛高。

---

## 6. 建议的推进路线（约 8–10 周）

1. **第 1–2 周：统一评测 + 复现 + 诊断。**
   - 基于 Time-Series-Library 或 TFB 搭一套统一评测：至少 3 个随机种子，`drop_last=False`，lookback 在 L ∈ {96, 336, 672} 中搜索。
   - 优先复现有开源代码的模型：iTransformer、TQNet、CycleNet、S-Mamba、GPS-Mamba，以及 DLinear、PatchTST、PAttn 等简单基线。
   - 同时把方向 ② 的诊断指标算出来。
2. **第 3–4 周：上界和对照小实验，选定方向。** 跑 §5 里各方向“1–2 周能做的验证”，只保留上界明显的 1–2 个方向。
3. **第 5–9 周：方法开发。** 包括消融、合成真值验证、效率评测（在 862 维的 Traffic 上测显存和延迟）。
4. **第 10 周起：写作。**
   - 方法型论文：可投 KDD、AAAI、IJCAI、ICDE、CIKM，或 Neurocomputing、ESWA（和库内多篇论文同一档次）。
   - 分析型论文：可投 NeurIPS Datasets & Benchmarks 或 TMLR。

## 7. 写论文时建议补齐的实验规范

库内多篇论文在这些地方吃亏，审稿人越来越看重：

- [ ] 至少 3 个随机种子，报均值和标准差；关键结论做显著性检验。
- [ ] 同时报告 L=96 和搜索后的最优 L；`drop_last=False`；基线**自己重跑**，不要直接抄数字。
- [ ] 包含强而简单的基线：DLinear、PatchTST、CycleNet、TQNet、PAttn、iTransformer、S-Mamba。
- [ ] 按耦合强度分组报告：强耦合组（PEMS、SD、Solar、Traffic）和弱耦合组（ETT、ECL、Weather）。
- [ ] 凡是声称“学到了动态/稀疏依赖”，都用带真值的合成数据验证结构恢复，报 AUROC 或 F1。
- [ ] 效率指标报全：参数量、FLOPs、延迟、显存，至少在一个高维数据集（Traffic）上测。

---

## 8. 参考链接

**库内论文**：见 §1（PDF 都在仓库根目录）。

**外部工作**（2025–2026 年的条目来自联网检索，只看了摘要）

- **跨变量 / 稀疏 / MoE**
  - MS-FLOW: https://arxiv.org/abs/2605.08289
  - Li-Net (ICDE'26): https://arxiv.org/abs/2603.18712
  - TimeFilter: https://arxiv.org/html/2501.13041v1
  - PatchMoE: https://arxiv.org/abs/2509.22279v2
  - Dynamic TMoE: https://arxiv.org/pdf/2605.20678
  - TQNet: https://arxiv.org/abs/2505.12917
- **置换不变 / TSFM 的跨通道能力**
  - CPiRi (ICLR'26): https://arxiv.org/abs/2601.20318
  - Channel Adapter for TSFMs (ICML'26): https://proceedings.mlr.press/v306/li26bu.html
  - Chronos-2: https://arxiv.org/pdf/2510.15821
  - TSFM 在协变量不确定下的负荷预测基准: https://arxiv.org/pdf/2610.07232
- **SSM 扫描顺序**
  - MambaTS: https://arxiv.org/abs/2405.16440
- **测试时扩展 / 推理 / 适应**
  - Time-R1: https://arxiv.org/abs/2506.10630
  - 测试时高效的预训练模型组合（ICLR'26，二手评述页）: https://liner.com/review/testtime-efficient-pretrained-model-portfolios-for-time-series-forecasting
  - COSA (ICLR'26): https://mlanthology.org/iclr/2026/im2026iclr-cosa/
  - PETSA: https://arxiv.org/html/2506.23424v1
  - AdaNODEs: https://arxiv.org/abs/2601.12893v1
- **文本 / 上下文辅助预测**
  - TimesX: https://www.alphaxiv.org/abs/2607.06973
  - What If TSF: https://arxiv.org/abs/2601.08509v1
  - Dr-CiK: https://www.alphaxiv.org/abs/2605.27904
  - PostTime: https://arxiv.org/html/2605.29401v1
  - Augur (ACL'26): https://aclanthology.org/2026.acl-long.32/
- **长上下文**
  - Timer-XL: https://arxiv.org/html/2410.04803v4
  - Timer-S1: https://arxiv.org/pdf/2603.04791.pdf
  - StreamTimer (IJCAI'26): https://www.ijcai.org/proceedings/2026/0327.pdf
- **评测**
  - TFB: https://arxiv.org/html/2403.20150v4
  - No champions (TMLR'26): https://arxiv.org/abs/2502.14045
  - It's TIME: https://arxiv.org/pdf/2602.12147
  - 关于 channel dependence 与 lookback 偏差的研究（二手评述页）: https://www.themoonlight.io/es/review/channel-dependence-limited-lookback-windows-and-the-simplicity-of-datasets-how-biased-is-time-series-forecasting
- **不规则 / 概率预测**
  - ASTGI: https://arxiv.org/pdf/2509.23313v3
  - ReIMTS: https://arxiv.org/abs/2602.21498
  - EnTransformer: https://www.alphaxiv.org/abs/2603.11909
