# 真实数据验证：变量混合层的改进（ETTh1 / Weather / Electricity）

本目录在三个公开基准上，检验几种针对"变量相关性建模"的改进是否真的有效。所有模型共用同一个 iTransformer / S-Mamba 式骨架，只替换变量混合层（下称 VC 层）或加入单一改动；每种配置跑 3 个种子，按种子配对比较。

> 这是 CPU 上的受控对比：模型宽度、层数和训练步数都比官方配置小（见第 3 节），所以绝对误差不能直接和论文表格比较，结论只针对"同一预算下改动前后的差异"。

## 结论速览

**按事先定好的判据（每个种子都更好，且平均改善 ≥ 1%）：**

| 改进 | ETTh1（7 个变量） | Weather（21） | Electricity（321） |
|---|---|---|---|
| H1 稀疏 VC（检索后注意）替代注意力 | 微弱改善 −0.5% | **有效 −2.0%** | 变差 +0.7% |
| H1 稀疏 VC（动态 top-K）替代注意力 | 不确定 | 变差 +1.4% | 变差 +0.8% |
| H1 稀疏 VC 替代双向 Mamba | 有效 −1.1% | 不确定 / 变差 | 变差 +3.8% ~ +3.9% |
| **H2 身份嵌入（双向 Mamba）** | 不确定（96 步）/ 变差 +1.3%（336 步） | **有效 −5.5%（96 步）/ −3.1%（336 步）** | **有效 −3.9%** |
| **H2 身份嵌入（注意力）** | 不确定 | **有效 −5.0%（96 步）/ −3.2%（336 步）** | **有效 −3.3%** |
| H3 打乱变量顺序（双向 Mamba） | 3 个种子都变好 0.6% ~ 1.2% | 有好有坏（−0.9% ~ +1.9%） | 3 个种子都变差 1.2% ~ 1.5% |
| H4 随机顺序训练 + 4 顺序平均（双向 Mamba） | **有效 −2.2%** | 变差 +2.2% | 变差 +1.1% |

1. **确认有效的改进只有一个：给变量 token 加可学习的身份嵌入（H2）。** 在两个多变量数据集上，对注意力、双向 Mamba、检索后注意三种骨架都有效，每个种子都更好，并且在 Weather 上从 96 步到 336 步都成立。在只有 7 个变量的 ETTh1 上无效，336 步时对双向 Mamba 还有害，应只在变量较多、相关性较强的数据上使用。这个做法本身已有先例（STID、EMAformer、ss-Mamba），这里验证的是它对 S-Mamba 式模型稳定有效。
2. **最好的组合是双向 Mamba + 身份嵌入**：Electricity 0.1822（比 S-Mamba 式低 3.9%，比 iTransformer 式低 6.8%），Weather 0.1749（分别低 5.5%、7.4%）。在 ETTh1 上，最好的是加了随机顺序训练和 4 顺序平均的版本（0.3784）：比 CI 每个种子都更好，但平均只好 0.8%，属于微弱改善。
3. **合成数据上"稀疏胜过稠密"的结论没有迁移到真实数据（H1）。** Electricity 上两种稀疏 VC 都比注意力和双向 Mamba 差；只在 Weather 上，检索后注意比注意力好 2%，但也没超过双向 Mamba。
4. **双向 Mamba 确实对变量顺序敏感（H3），但"让它与顺序无关"并不是好的修正（H4）。** Electricity 和 Weather 的原始变量顺序本身带有有用的结构：打乱后变差，强迫顺序无关也变差。测试时对多个顺序取平均相对单一随机顺序稳定有益（ETTh1 −1.1%，Electricity −1.35%），但补不回固定顺序的优势。改进方向应是"找到或学到好的顺序"（GPS-Mamba、MambaTS 的思路），而不是忽略顺序。

---

## 1. 检索：现有模型还有哪些值得改进的地方

| 问题 | 证据（论文与社区） | 对应的改进 | 本目录是否实验 |
|---|---|---|---|
| **Mamba 沿变量维度扫描，结果依赖人为指定的变量顺序** | S-Mamba 原文称对顺序不敏感，但只在 Electricity 上挪动了部分变量；MambaTS、SOR-Mamba、CRS-Mamba、GPS-Mamba 都认为顺序重要并各自给出解法 | 训练时随机打乱扫描顺序（MambaTS 的 VPT），测试时对多个顺序取平均 | 是（H3、H4） |
| **稠密的跨变量混合会引入无关变量的噪声** | MS-FLOW（2026，按样本取 top-K 稀疏依赖，报告比 iTransformer 低约 9%）、TimeFilter（ICML 2025）、DUET、DiAG（2026，指出 iTransformer 上收益不稳定） | 稀疏 VC 层：按内容取 top-K，或先按嵌入检索候选再注意 | 是（H1） |
| **变量 token 之间难以区分** | STID（CIKM 2022）发现身份嵌入是多变量预测的关键；EMAformer（AAAI 2026）给 iTransformer 加嵌入；ss-Mamba（2025）用语义身份嵌入 | 给每个变量 token 加可学习的身份嵌入 | 是（H2） |
| **评测口径不一、提升常在噪声内** | TFB（VLDB 2024）指出 drop_last 等不公平比较；"There Are No Champions"（TMLR 2026）发现调参后 SOTA 差距消失；2026 年的 Mamba 时序综述指出 Mamba 模型的结果难以同口径比较；TIME（ICML 2026）、fev-bench、QuitoBench 等新基准关注泄漏与数据质量 | 同一骨架、同一预算、多种子配对、只用验证集选模、评估全部测试窗口 | 作为实验规范 |
| 其他 | FreDF（ICLR 2025）指出直接多步预测的标签自相关偏差；归一化与非平稳性处理 | 频域损失等 | 否（留作后续） |

## 2. 假设与判定标准（实验前设定）

- **H1 稀疏替代稠密**：在多变量数据上，用稀疏 VC 层替换注意力或双向 Mamba 能降低误差。
- **H2 身份嵌入**：给变量 token 加可学习身份嵌入能降低误差。
- **H3 顺序敏感**：打乱变量顺序会改变双向 Mamba 的结果，且差异超过种子间波动。
- **H4 顺序鲁棒**：随机顺序训练、测试时多顺序平均，能降低双向 Mamba 的误差。

**判定**：用验证集做早停与选模；候选在每个种子上都比参照低、且配对平均改善至少 1%，才记为"有效"；每个种子都更低但不足 1% 记为"微弱改善"；每个种子都更高记为"变差"；其余为"不确定"。

## 3. 实验设置（`data_real.py`、`models_real.py`、`train_real.py`）

- **数据**：ETTh1（7 个变量）、Weather（21）、Electricity（321），标准划分（ETT 为 12/4/4 个月，其余 70/10/20），用训练段统计量标准化；回看 96 步，预测 96 步；测试时评估全部窗口，不丢弃 batch。Traffic（862 个变量）在 CPU 上太慢，没有纳入；PEMS 系列在所用镜像中没有。
- **骨架**：实例归一化 → 每个变量的 96 步历史经线性层变成一个 token（D=64）→ 2 层 [VC 层 + FFN，均为残差 + LayerNorm] → 线性层输出 96 步 → 反归一化。
- **VC 层**：

| 键名 | 说明 |
|---|---|
| `none` | 不做跨变量混合（通道独立） |
| `attn` | 8 头自注意力（iTransformer 式） |
| `bimamba` | 前向 + 反向两个 Mamba 块相加（S-Mamba 式；expand=1，d_conv=2，d_state=2） |
| `dyntopk` | 对全部变量计算内容分数 QKᵀ，每行只保留 top-K 再 softmax（MS-FLOW 式，K=16，单头；不是 MS-FLOW 的完整复现） |
| `retrieve` | 先在可学习嵌入空间里为每个变量检索 k=16 个候选，再按内容 + 嵌入相似度加权（合成实验中的稀疏项） |
| `+id` | 在变量 token 上加可学习身份嵌入（初始化为 0） |
| `bimamba-shuffled` | 双向 Mamba，但按一个固定的随机顺序扫描变量 |
| `bimamba-vpt` | 双向 Mamba，每个训练步随机打乱扫描顺序；测试时按原顺序 |
| `+tta4` | 同一个训练好的模型，测试时把每层 VC 输出在 4 个固定随机顺序上取平均 |

- **训练**：Adam，学习率 5e-4，余弦退火，batch 32；每"轮"随机抽 150 个 batch，最多 10 轮，验证 MSE 连续 3 轮不降即停，取验证最优的权重做测试；种子 0/1/2；CPU 单线程。
- **与官方配置的差异**：官方 S-Mamba 在 Electricity 上用 d_model=512、3 层、d_state=16，在 Weather、ETTh1 上用 d_state=2；这里统一用 D=64、2 层、d_state=2，训练量约为官方的四分之一，也没有加入时间特征 token。双向 Mamba 用纯 PyTorch 的分块并行扫描实现，与逐步递推的数值误差约 1e-6。
- **流程核对**：ETTh1 上注意力版和双向 Mamba 版的测试 MSE/MAE 分别为 0.386/0.404 和 0.387/0.406，与论文报告的 iTransformer（0.386/0.405）和 S-Mamba（0.386/0.405）基本一致。

## 4. 结果

### 预测 96 步

#### ETTh1（预测 96 步）

| 模型 | 测试 MSE | 测试 MAE | 种子数 | 平均耗时 |
|---|---|---|---|---|
| CI（无变量混合） | 0.3815 ± 0.0017 | 0.3988 ± 0.0015 | 3 | 0.1 min |
| 注意力（iTransformer 式） | 0.3849 ± 0.0012 | 0.4033 ± 0.0003 | 3 | 0.3 min |
| 双向 Mamba（S-Mamba 式） | 0.3872 ± 0.0009 | 0.4062 ± 0.0009 | 3 | 0.6 min |
| 动态 top-K（MS-FLOW 式） | 0.3831 ± 0.0014 | 0.4015 ± 0.0009 | 3 | 0.2 min |
| 检索后注意（候选来自嵌入） | 0.3830 ± 0.0011 | 0.4023 ± 0.0008 | 3 | 0.3 min |
| 注意力 + 身份嵌入 | 0.3847 ± 0.0011 | 0.4034 ± 0.0003 | 3 | 0.3 min |
| 双向 Mamba + 身份嵌入 | 0.3872 ± 0.0010 | 0.4060 ± 0.0009 | 3 | 0.6 min |
| 检索后注意 + 身份嵌入 | 0.3827 ± 0.0012 | 0.4021 ± 0.0008 | 3 | 0.3 min |
| 双向 Mamba（变量顺序打乱） | 0.3838 ± 0.0003 | 0.4030 ± 0.0003 | 3 | 0.6 min |
| 双向 Mamba + 随机顺序训练 | 0.3828 ± 0.0017 | 0.4019 ± 0.0013 | 3 | 0.8 min |
| 双向 Mamba + 随机顺序训练 + 4 顺序平均 | 0.3787 ± 0.0016 | 0.3984 ± 0.0010 | 3 | 0.8 min |
| 双向 Mamba + 身份嵌入 + 随机顺序训练 | 0.3825 ± 0.0017 | 0.4015 ± 0.0014 | 3 | 0.8 min |
| 双向 Mamba + 身份嵌入 + 随机顺序训练 + 4 顺序平均 | 0.3784 ± 0.0017 | 0.3980 ± 0.0011 | 3 | 0.8 min |

#### Electricity（预测 96 步）

| 模型 | 测试 MSE | 测试 MAE | 种子数 | 平均耗时 |
|---|---|---|---|---|
| CI（无变量混合） | 0.1998 ± 0.0004 | 0.2831 ± 0.0007 | 3 | 3.7 min |
| 注意力（iTransformer 式） | 0.1954 ± 0.0004 | 0.2813 ± 0.0007 | 3 | 46.1 min |
| 双向 Mamba（S-Mamba 式） | 0.1895 ± 0.0004 | 0.2782 ± 0.0005 | 3 | 36.1 min |
| 动态 top-K（MS-FLOW 式） | 0.1969 ± 0.0009 | 0.2822 ± 0.0011 | 3 | 13.2 min |
| 检索后注意（候选来自嵌入） | 0.1967 ± 0.0003 | 0.2815 ± 0.0004 | 3 | 21.4 min |
| 注意力 + 身份嵌入 | 0.1890 ± 0.0008 | 0.2790 ± 0.0008 | 3 | 47.0 min |
| 双向 Mamba + 身份嵌入 | 0.1822 ± 0.0003 | 0.2751 ± 0.0006 | 3 | 36.4 min |
| 检索后注意 + 身份嵌入 | 0.1914 ± 0.0001 | 0.2798 ± 0.0006 | 3 | 21.7 min |
| 双向 Mamba（变量顺序打乱） | 0.1921 ± 0.0007 | 0.2787 ± 0.0006 | 3 | 36.4 min |
| 双向 Mamba + 随机顺序训练 | 0.1943 ± 0.0010 | 0.2801 ± 0.0007 | 3 | 36.1 min |
| 双向 Mamba + 随机顺序训练 + 4 顺序平均 | 0.1916 ± 0.0006 | 0.2779 ± 0.0006 | 3 | 36.1 min |
| 双向 Mamba + 身份嵌入 + 随机顺序训练 | 0.1870 ± 0.0008 | 0.2780 ± 0.0006 | 3 | 35.8 min |
| 双向 Mamba + 身份嵌入 + 随机顺序训练 + 4 顺序平均 | 0.1834 ± 0.0007 | 0.2750 ± 0.0006 | 3 | 35.8 min |

#### Weather（预测 96 步）

| 模型 | 测试 MSE | 测试 MAE | 种子数 | 平均耗时 |
|---|---|---|---|---|
| CI（无变量混合） | 0.1933 ± 0.0002 | 0.2336 ± 0.0001 | 3 | 0.3 min |
| 注意力（iTransformer 式） | 0.1889 ± 0.0010 | 0.2298 ± 0.0003 | 3 | 0.7 min |
| 双向 Mamba（S-Mamba 式） | 0.1850 ± 0.0010 | 0.2260 ± 0.0009 | 3 | 1.9 min |
| 动态 top-K（MS-FLOW 式） | 0.1915 ± 0.0018 | 0.2314 ± 0.0011 | 3 | 0.5 min |
| 检索后注意（候选来自嵌入） | 0.1852 ± 0.0009 | 0.2284 ± 0.0014 | 3 | 1.4 min |
| 注意力 + 身份嵌入 | 0.1795 ± 0.0010 | 0.2242 ± 0.0005 | 3 | 0.7 min |
| 双向 Mamba + 身份嵌入 | 0.1749 ± 0.0006 | 0.2188 ± 0.0007 | 3 | 1.9 min |
| 检索后注意 + 身份嵌入 | 0.1777 ± 0.0009 | 0.2230 ± 0.0011 | 3 | 1.3 min |
| 双向 Mamba（变量顺序打乱） | 0.1856 ± 0.0018 | 0.2270 ± 0.0006 | 3 | 1.9 min |
| 双向 Mamba + 随机顺序训练 | 0.1894 ± 0.0004 | 0.2287 ± 0.0007 | 3 | 1.9 min |
| 双向 Mamba + 随机顺序训练 + 4 顺序平均 | 0.1891 ± 0.0007 | 0.2288 ± 0.0010 | 3 | 1.9 min |
| 双向 Mamba + 身份嵌入 + 随机顺序训练 | 0.1760 ± 0.0014 | 0.2199 ± 0.0013 | 3 | 2.0 min |
| 双向 Mamba + 身份嵌入 + 随机顺序训练 + 4 顺序平均 | 0.1760 ± 0.0016 | 0.2198 ± 0.0015 | 3 | 2.0 min |

### 预测 336 步（身份嵌入的跨长度检验）

#### ETTh1（预测 336 步）

| 模型 | 测试 MSE | 测试 MAE | 种子数 | 平均耗时 |
|---|---|---|---|---|
| 注意力（iTransformer 式） | 0.4740 ± 0.0013 | 0.4543 ± 0.0012 | 3 | 0.2 min |
| 双向 Mamba（S-Mamba 式） | 0.4856 ± 0.0057 | 0.4653 ± 0.0028 | 3 | 0.4 min |
| 注意力 + 身份嵌入 | 0.4732 ± 0.0022 | 0.4537 ± 0.0016 | 3 | 0.2 min |
| 双向 Mamba + 身份嵌入 | 0.4918 ± 0.0048 | 0.4681 ± 0.0029 | 3 | 0.4 min |

#### Weather（预测 336 步）

| 模型 | 测试 MSE | 测试 MAE | 种子数 | 平均耗时 |
|---|---|---|---|---|
| 注意力（iTransformer 式） | 0.2896 ± 0.0004 | 0.3063 ± 0.0004 | 3 | 0.6 min |
| 双向 Mamba（S-Mamba 式） | 0.2864 ± 0.0010 | 0.3042 ± 0.0004 | 3 | 1.7 min |
| 注意力 + 身份嵌入 | 0.2803 ± 0.0006 | 0.3012 ± 0.0004 | 3 | 0.7 min |
| 双向 Mamba + 身份嵌入 | 0.2774 ± 0.0004 | 0.2986 ± 0.0004 | 3 | 1.7 min |


### 假设检验（逐种子配对）

预测 96 步：

| 假设 | 数据集 | 候选 vs 参照 | MSE 变化（配对平均） | 更好的种子 | 判定 |
|---|---|---|---|---|---|
| H1 稀疏替代注意力 | ETTh1 | 动态 top-K（MS-FLOW 式） vs 注意力（iTransformer 式） | -0.47% | 2/3 | 不确定 |
| H1 稀疏替代注意力 | ETTh1 | 检索后注意（候选来自嵌入） vs 注意力（iTransformer 式） | -0.48% | 3/3 | 微弱改善（<1%） |
| H1 稀疏替代双向 Mamba | ETTh1 | 动态 top-K（MS-FLOW 式） vs 双向 Mamba（S-Mamba 式） | -1.06% | 3/3 | 有效 |
| H1 稀疏替代双向 Mamba | ETTh1 | 检索后注意（候选来自嵌入） vs 双向 Mamba（S-Mamba 式） | -1.08% | 3/3 | 有效 |
| H2 身份嵌入 | ETTh1 | 注意力 + 身份嵌入 vs 注意力（iTransformer 式） | -0.04% | 2/3 | 不确定 |
| H2 身份嵌入 | ETTh1 | 双向 Mamba + 身份嵌入 vs 双向 Mamba（S-Mamba 式） | +0.02% | 1/3 | 不确定 |
| H2 身份嵌入 | ETTh1 | 检索后注意 + 身份嵌入 vs 检索后注意（候选来自嵌入） | -0.08% | 3/3 | 微弱改善（<1%） |
| H4 顺序鲁棒 | ETTh1 | 双向 Mamba + 随机顺序训练 vs 双向 Mamba（S-Mamba 式） | -1.14% | 3/3 | 有效 |
| H4 顺序鲁棒 | ETTh1 | 双向 Mamba + 随机顺序训练 + 4 顺序平均 vs 双向 Mamba（S-Mamba 式） | -2.19% | 3/3 | 有效 |
| H4 顺序鲁棒 | ETTh1 | 双向 Mamba + 随机顺序训练 + 4 顺序平均 vs 双向 Mamba + 随机顺序训练 | -1.06% | 3/3 | 有效 |
| H4 顺序鲁棒 | ETTh1 | 双向 Mamba + 身份嵌入 + 随机顺序训练 + 4 顺序平均 vs 双向 Mamba + 身份嵌入 | -2.28% | 3/3 | 有效 |
| 组合 vs 原版 | ETTh1 | 双向 Mamba + 身份嵌入 + 随机顺序训练 + 4 顺序平均 vs 双向 Mamba（S-Mamba 式） | -2.26% | 3/3 | 有效 |
| 组合 vs 原版 | ETTh1 | 双向 Mamba + 身份嵌入 + 随机顺序训练 + 4 顺序平均 vs 注意力（iTransformer 式） | -1.68% | 3/3 | 有效 |
| 组合 vs 原版 | ETTh1 | 双向 Mamba + 身份嵌入 vs 注意力（iTransformer 式） | +0.62% | 0/3 | 变差 |
| 参考：变量混合 vs CI | ETTh1 | 注意力（iTransformer 式） vs CI（无变量混合） | +0.89% | 0/3 | 变差 |
| 参考：变量混合 vs CI | ETTh1 | 双向 Mamba（S-Mamba 式） vs CI（无变量混合） | +1.50% | 0/3 | 变差 |
| 参考：变量混合 vs CI | ETTh1 | 检索后注意（候选来自嵌入） vs CI（无变量混合） | +0.40% | 0/3 | 变差 |
| H1 稀疏替代注意力 | Electricity | 动态 top-K（MS-FLOW 式） vs 注意力（iTransformer 式） | +0.78% | 0/3 | 变差 |
| H1 稀疏替代注意力 | Electricity | 检索后注意（候选来自嵌入） vs 注意力（iTransformer 式） | +0.65% | 0/3 | 变差 |
| H1 稀疏替代双向 Mamba | Electricity | 动态 top-K（MS-FLOW 式） vs 双向 Mamba（S-Mamba 式） | +3.90% | 0/3 | 变差 |
| H1 稀疏替代双向 Mamba | Electricity | 检索后注意（候选来自嵌入） vs 双向 Mamba（S-Mamba 式） | +3.76% | 0/3 | 变差 |
| H2 身份嵌入 | Electricity | 注意力 + 身份嵌入 vs 注意力（iTransformer 式） | -3.29% | 3/3 | 有效 |
| H2 身份嵌入 | Electricity | 双向 Mamba + 身份嵌入 vs 双向 Mamba（S-Mamba 式） | -3.87% | 3/3 | 有效 |
| H2 身份嵌入 | Electricity | 检索后注意 + 身份嵌入 vs 检索后注意（候选来自嵌入） | -2.69% | 3/3 | 有效 |
| H4 顺序鲁棒 | Electricity | 双向 Mamba + 随机顺序训练 vs 双向 Mamba（S-Mamba 式） | +2.50% | 0/3 | 变差 |
| H4 顺序鲁棒 | Electricity | 双向 Mamba + 随机顺序训练 + 4 顺序平均 vs 双向 Mamba（S-Mamba 式） | +1.11% | 0/3 | 变差 |
| H4 顺序鲁棒 | Electricity | 双向 Mamba + 随机顺序训练 + 4 顺序平均 vs 双向 Mamba + 随机顺序训练 | -1.35% | 3/3 | 有效 |
| H4 顺序鲁棒 | Electricity | 双向 Mamba + 身份嵌入 + 随机顺序训练 + 4 顺序平均 vs 双向 Mamba + 身份嵌入 | +0.66% | 0/3 | 变差 |
| 组合 vs 原版 | Electricity | 双向 Mamba + 身份嵌入 + 随机顺序训练 + 4 顺序平均 vs 双向 Mamba（S-Mamba 式） | -3.24% | 3/3 | 有效 |
| 组合 vs 原版 | Electricity | 双向 Mamba + 身份嵌入 + 随机顺序训练 + 4 顺序平均 vs 注意力（iTransformer 式） | -6.14% | 3/3 | 有效 |
| 组合 vs 原版 | Electricity | 双向 Mamba + 身份嵌入 vs 注意力（iTransformer 式） | -6.76% | 3/3 | 有效 |
| 参考：变量混合 vs CI | Electricity | 注意力（iTransformer 式） vs CI（无变量混合） | -2.20% | 3/3 | 有效 |
| 参考：变量混合 vs CI | Electricity | 双向 Mamba（S-Mamba 式） vs CI（无变量混合） | -5.14% | 3/3 | 有效 |
| 参考：变量混合 vs CI | Electricity | 检索后注意（候选来自嵌入） vs CI（无变量混合） | -1.57% | 3/3 | 有效 |
| H1 稀疏替代注意力 | Weather | 动态 top-K（MS-FLOW 式） vs 注意力（iTransformer 式） | +1.38% | 0/3 | 变差 |
| H1 稀疏替代注意力 | Weather | 检索后注意（候选来自嵌入） vs 注意力（iTransformer 式） | -1.96% | 3/3 | 有效 |
| H1 稀疏替代双向 Mamba | Weather | 动态 top-K（MS-FLOW 式） vs 双向 Mamba（S-Mamba 式） | +3.51% | 0/3 | 变差 |
| H1 稀疏替代双向 Mamba | Weather | 检索后注意（候选来自嵌入） vs 双向 Mamba（S-Mamba 式） | +0.11% | 1/3 | 不确定 |
| H2 身份嵌入 | Weather | 注意力 + 身份嵌入 vs 注意力（iTransformer 式） | -4.98% | 3/3 | 有效 |
| H2 身份嵌入 | Weather | 双向 Mamba + 身份嵌入 vs 双向 Mamba（S-Mamba 式） | -5.49% | 3/3 | 有效 |
| H2 身份嵌入 | Weather | 检索后注意 + 身份嵌入 vs 检索后注意（候选来自嵌入） | -4.05% | 3/3 | 有效 |
| H4 顺序鲁棒 | Weather | 双向 Mamba + 随机顺序训练 vs 双向 Mamba（S-Mamba 式） | +2.36% | 0/3 | 变差 |
| H4 顺序鲁棒 | Weather | 双向 Mamba + 随机顺序训练 + 4 顺序平均 vs 双向 Mamba（S-Mamba 式） | +2.22% | 0/3 | 变差 |
| H4 顺序鲁棒 | Weather | 双向 Mamba + 随机顺序训练 + 4 顺序平均 vs 双向 Mamba + 随机顺序训练 | -0.14% | 2/3 | 不确定 |
| H4 顺序鲁棒 | Weather | 双向 Mamba + 身份嵌入 + 随机顺序训练 + 4 顺序平均 vs 双向 Mamba + 身份嵌入 | +0.65% | 0/3 | 变差 |
| 组合 vs 原版 | Weather | 双向 Mamba + 身份嵌入 + 随机顺序训练 + 4 顺序平均 vs 双向 Mamba（S-Mamba 式） | -4.88% | 3/3 | 有效 |
| 组合 vs 原版 | Weather | 双向 Mamba + 身份嵌入 + 随机顺序训练 + 4 顺序平均 vs 注意力（iTransformer 式） | -6.85% | 3/3 | 有效 |
| 组合 vs 原版 | Weather | 双向 Mamba + 身份嵌入 vs 注意力（iTransformer 式） | -7.44% | 3/3 | 有效 |
| 参考：变量混合 vs CI | Weather | 注意力（iTransformer 式） vs CI（无变量混合） | -2.28% | 3/3 | 有效 |
| 参考：变量混合 vs CI | Weather | 双向 Mamba（S-Mamba 式） vs CI（无变量混合） | -4.29% | 3/3 | 有效 |
| 参考：变量混合 vs CI | Weather | 检索后注意（候选来自嵌入） vs CI（无变量混合） | -4.19% | 3/3 | 有效 |

预测 336 步：

| 假设 | 数据集 | 候选 vs 参照 | MSE 变化（配对平均） | 更好的种子 | 判定 |
|---|---|---|---|---|---|
| H2 身份嵌入 | ETTh1 | 注意力 + 身份嵌入 vs 注意力（iTransformer 式） | -0.16% | 2/3 | 不确定 |
| H2 身份嵌入 | ETTh1 | 双向 Mamba + 身份嵌入 vs 双向 Mamba（S-Mamba 式） | +1.29% | 0/3 | 变差 |
| 组合 vs 原版 | ETTh1 | 双向 Mamba + 身份嵌入 vs 注意力（iTransformer 式） | +3.76% | 0/3 | 变差 |
| H2 身份嵌入 | Weather | 注意力 + 身份嵌入 vs 注意力（iTransformer 式） | -3.22% | 3/3 | 有效 |
| H2 身份嵌入 | Weather | 双向 Mamba + 身份嵌入 vs 双向 Mamba（S-Mamba 式） | -3.14% | 3/3 | 有效 |
| 组合 vs 原版 | Weather | 双向 Mamba + 身份嵌入 vs 注意力（iTransformer 式） | -4.22% | 3/3 | 有效 |

### 变量顺序敏感性（H3）

| 数据集 | 原顺序 MSE | 打乱顺序 MSE | 逐种子差异 | 原顺序的种子间标准差 |
|---|---|---|---|---|
| ETTh1 | 0.3872 ± 0.0009 | 0.3838 ± 0.0003 | -0.83%, -1.16%, -0.62% | 0.0009 |
| Electricity | 0.1895 ± 0.0004 | 0.1921 ± 0.0007 | +1.37%, +1.48%, +1.16% | 0.0004 |
| Weather | 0.1850 ± 0.0010 | 0.1856 ± 0.0018 | -0.16%, +1.90%, -0.86% | 0.0010 |

逐种子差异是"打乱顺序 / 原顺序 − 1"。Electricity 上 3 个种子都变差 1.2%–1.5%，约为种子间标准差的 3 倍；ETTh1 上 3 个种子都变好；Weather 上方向不一致，最大变化 1.9%。S-Mamba 原文称性能对变量顺序不敏感，这里的结果不支持这一点。

## 5. 结论

1. **身份嵌入（H2）是本轮唯一稳定有效的改进**，适用于变量较多、相关性较强的数据（Weather、Electricity），对三种 VC 层都有效，并在 Weather 的 336 步上复现。原因可能在于：变量 token 只由一段 96 步的历史线性映射而来，不同变量的历史可能很像，加入身份后，FFN 和 VC 层能学到"这个变量自己的规律"以及"它通常和谁相关"。这与 STID"样本在空间维度上难以区分"的论点一致。代价是模型无法直接泛化到训练时没见过的变量。
2. **稀疏 VC（H1）在合成数据上的大幅优势没有在真实数据上复现。** 合成数据的"每个变量只依赖本簇的领头变量"结构，比 Electricity 这类真实数据稀疏得多；真实数据里稠密的混合（尤其是双向 Mamba）反而更好。这也提醒：合成实验适合排除坏设计，不能代替真实数据上的结论。
3. **关于 Mamba 的变量顺序（H3、H4）**：顺序确实影响结果，但影响方向依数据集而定。原始顺序有意义时（Electricity、Weather），保留固定顺序优于随机顺序训练；原始顺序不好时（ETTh1），随机顺序训练加多顺序平均有效。值得继续做的是"可学习或数据驱动的顺序"，并与多顺序平均结合。
4. **双向 Mamba 是三个数据集中最强的 VC 层之一**：在 Electricity 和 Weather 上，相对注意力分别低 3.0% 和 2.1%，与 S-Mamba 论文的结论一致；但在 ETTh1 上不如通道独立。

## 6. 局限

- 主实验只有预测 96 步；336 步只在 Weather 和 ETTh1 上检验了身份嵌入，Electricity 的 336 步没有跑。只有 3 个数据集，最多 321 个变量；Traffic 和 PEMS 没有覆盖。
- 模型和训练预算都比官方配置小，双向 Mamba 在 Electricity 上用的 d_state=2 小于官方的 16；更大模型下各改动的幅度可能不同。
- 每种配置 3 个种子；"有效"的判定依赖配对比较，没有做更多种子的显著性检验。
- `dyntopk` 只是 MS-FLOW 的"按内容取 top-K"机制，没有复现它的 patch 级时间混合和其他设计，所以 H1 的否定结论不等于否定 MS-FLOW。

## 7. 复现

数据：从 [Hugging Face 上的 Time-Series-Library 数据镜像](https://huggingface.co/datasets/thuml/Time-Series-Library) 下载 `ETT-small/ETTh1.csv`、`weather/weather.csv`、`electricity/electricity.csv` 到同一目录 `DATA`。需要 PyTorch、pandas、numpy。

```bash
python train_real.py --data-dir DATA --data Electricity --models attn,bimamba,bimamba+id --seeds 0 1 2
python train_real.py --data-dir DATA --data Weather --models bimamba-vpt,bimamba+id-vpt --seeds 0 1 2
# 批量运行（本目录的全部结果）：
cat jobs_T96.txt jobs_T96_batch2.txt jobs_T336.txt | xargs -P 4 -L 1 sh -c 'python train_real.py --data-dir DATA "$0" "$@" >> logs/results.log'
python summarize_real.py logs/results_T96.log      # 或 logs/results_T336.log
```

| 文件 | 内容 |
|---|---|
| `data_real.py` | 数据读取与标准划分 |
| `models_real.py` | 骨架与各 VC 层（含纯 PyTorch 的 Mamba 块） |
| `train_real.py` | 训练、早停与测试；`VARIANTS` 列出全部键名 |
| `summarize_real.py` | 生成结果表与假设检验 |
| `jobs_T96.txt`、`jobs_T96_batch2.txt`、`jobs_T336.txt` | 三批任务列表 |
| `logs/results_T96.log`、`logs/results_T336.log` | 全部原始结果 |

## 8. 参考

- [S-Mamba](https://arxiv.org/abs/2403.11144)（仓库 #6）、[iTransformer](https://arxiv.org/abs/2310.06625)
- [MambaTS](https://arxiv.org/abs/2405.16440)、[SOR-Mamba](https://arxiv.org/abs/2410.23356)
- [MS-FLOW](https://arxiv.org/abs/2605.08289)、[TimeFilter](https://arxiv.org/abs/2501.13041)、[DUET](https://arxiv.org/abs/2412.10859)、[DiAG](https://www.mdpi.com/2078-2489/17/4/394)
- [STID](https://arxiv.org/abs/2208.05233)、[EMAformer](https://arxiv.org/abs/2511.08396)、[ss-Mamba](https://arxiv.org/abs/2506.14802)
- [TFB](https://arxiv.org/abs/2403.20150)、[There Are No Champions](https://mlanthology.org/tmlr/2026/brigato2026tmlr-there)、[TIME](https://arxiv.org/abs/2602.12147)、[QuitoBench](https://hq-bench.github.io/quito/)
- [FreDF](https://arxiv.org/abs/2402.02399)、[Janossy Pooling（对顺序取平均得到置换不变函数）](https://arxiv.org/abs/1811.01900)
