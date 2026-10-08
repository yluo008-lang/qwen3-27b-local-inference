---
layout: default
title: "PCIe Attached Memory 作为 KV Cache 层：单卡验证测试方案"
---

# PCIe Attached Memory 作为 KV Cache 层：单卡验证测试方案

**环境：** RTX 4080 16GB（PCIe Gen4 x16）· 主机内存 32GB · NVMe · Qwen3 27B 本地推理
**目标：** 在单卡上验证核心假设，做出可复用的后端和预测模型，再拿到服务器上补测多卡和 P2P。
**前提：** 第 4 步起改用 Linux（Ubuntu 双系统优先，WSL2 次选）。Windows 下测出的 PCIe 数字只作为下限参考。

---

## 一、核心假设与判定

| 编号 | 假设 | 判定为不成立的条件 |
|---|---|---|
| H1 | KV 大小 = 有 KV 的层数 × KV head 数 × head 维度 × 2 × 每元素字节数 × token 数 | 和 llama.cpp 日志对比，误差超过 10% |
| H2 | decode 每生成一个 token 要读一遍全部 KV，TPOT 随上下文长度线性增长 | TPOT 基本不随上下文变化 |
| H3 | PCIe 只能用于 KV 的换入/换出，不能作为 decode 时的工作集 | 按 PCIe 带宽推算，32K 上下文时 TPOT 仍然可以接受 |
| H4 | 换入 KV 比重算 prefill 快 10 倍以上 | 两者相差不到 10 倍 |
| H5 | 链路带宽决定分层方案的收益，存在一个“低于它就不如重算”的最低规格 | 测试的带宽范围内，TTFT 对带宽不敏感 |

---

## 二、执行步骤

### 步骤 1：核对 KV 容量（验证 H1）· Windows 下即可做
1. 固定 `-ngl 99`，在 `-c 8192 / 16384 / 32768` 和 `--cache-type-k/v f16 / q8_0 / q4_0` 的每种组合下各启动一次，从启动日志里记下 KV cache 的大小（MiB）。
2. 从 GGUF 元数据里读出层数、KV head 数、head 维度，以及是否混合了线性注意力层。
3. 用公式算理论值，和日志对比。每元素字节数：f16 取 2，q8_0 约 1.06，q4_0 约 0.56。
4. 再用 `--parallel 1/2/4` 测一遍。注意 `-c` 是所有 slot 共用的总长度。

> 重点：27B 的 IQ4_XS 权重约 14GB，加上标准结构下 32K 的 KV，显存会超过 16GB，和你实测的 15.46GB 对不上。要先确认这个模型是不是只有部分层带 KV。**后面所有的带宽账都以这一步的结果为准。**

### 步骤 2：decode 带宽和 PCIe 上限（验证 H2、H3）
1. 固定 `-ngl 99`、KV 用 Q4、输出 256 个 token。上下文取 1K、2K、4K、8K、16K、24K，每个点重复 3 次，记录 TPOT。
2. 拟合 **TPOT = a + b·L**，算出读 KV 的实际带宽 BW_KV = 每 token 的 KV 字节数 / b，和显存带宽 717GB/s 对比。
3. 用 `bandwidthTest --memory=pinned` 实测主机到 GPU 的带宽，预计 22–26GB/s。
4. 推算如果 decode 直接通过 PCIe 读 KV，TPOT 会是多少：TPOT_PCIe ≈ a + KV(L) / BW_PCIe。
5. **复查 Q8 异常**（5.5 tok/s、约 105W、GPU 利用率 99%）：
   - 看任务管理器里“共享 GPU 内存”有没有上涨。
   - 在 NVIDIA 控制面板里把 CUDA Sysmem Fallback Policy 设为 *Prefer No Sysmem Fallback*，再跑一次。
   - 如果确认是显存溢出，就用第 4 步的公式核对实际变慢的程度。这能作为“外部内存不能直接拿来扩显存”的实测证据。

### 步骤 3：换入和重算对比（验证 H4）
```bash
llama-server ... -c 32768 --slot-save-path /path/kvslots
# 1. 发送 8K/16K/32K 的长 prompt，记录 prefill 时间（这是重算基线）
curl -X POST "http://localhost:8080/slots/0?action=save"    -d '{"filename":"s32k.bin"}'
curl -X POST "http://localhost:8080/slots/0?action=erase"
curl -X POST "http://localhost:8080/slots/0?action=restore" -d '{"filename":"s32k.bin"}'
# 2. 再发同一个 prompt，确认命中缓存、没有重算
```
每个上下文长度测三种情况：**重算、冷恢复（从 NVMe 读）、热恢复（文件在内存缓存里）**。恢复耗时要拆成“读文件”和“拷到 GPU”两部分。
预期：32K 重算约 20 秒以上，热恢复在 0.1 秒量级，冷恢复约 0.5 秒。

### 步骤 4：平台检查脚本（第 3 步的演练）
写一个 `check_platform.sh`，检查以下内容：
- `lspci -vv`（看 LnkSta）、`nvidia-smi -q`：链路代数和宽度
- `nvidia-smi -q -d MEMORY`：BAR1 大小，确认 ReBAR 是否打开
- `nvidia-smi topo -m`、`lspci -tv`：拓扑
- `dmesg | grep -i iommu`、`lspci -vvv | grep -i acsctl`：IOMMU 和 ACS 状态

**产出：** 拿到任何服务器，跑一遍就能知道 P2P 能不能用。

### 步骤 5：裸 DMA Benchmark（第 4 步，价值最高）
把主机上的 pinned 内存当作“PCIe 内存”，用 CUDA 或 PyTorch 加 CUDA event 计时。

| 测试 | 参数 | 要得到的结论 |
|---|---|---|
| block 大小扫描 | 4KB 到 64MB，分别测 pinned 和 pageable | 带宽在多大的 block 上饱和，KV block 落在曲线的哪个位置 |
| 小块与合并 | 1000 个 32KB 的块逐个拷贝，对比合并成一次 | 单次拷贝的固定开销 t₀ |
| 方向 | H2D、D2H、双向同时传 | 换入和换出能否并行 |
| 队列深度 | 1、2、4、8 个 stream | 并发能否提高利用率 |
| 拷贝和计算重叠 | 一个 stream 做 matmul，同时另一个拷贝 | 重叠率，以及两者是否互相拖慢 |
| NVMe 路径 | `fio`，再测“文件 → pinned 内存 → GPU”整条路径 | NVMe 层的实际带宽和延迟 |

**产出：** 拷贝耗时模型 **T(size) = t₀ + size / BW**，结果输出成 CSV。

### 步骤 6：模拟带宽争抢（第 5 步的单卡替代）
1. 2 到 4 个进程同时做 H2D 拷贝，记录总带宽和单个进程的 P99。这模拟两张 GPU 共用一条上行链路。
2. 用 STREAM 或 `mbw` 跑满 DDR 带宽，同时做 H2D 拷贝，看 PCIe 拷贝会不会因此变慢。
3. 建一个排队模型，参数为 N 张 GPU、每个 Switch 挂 k 张、上行链路宽 W、换入请求速率 λ，输出总带宽和 P99。

### 步骤 7：做可限速的模拟 PCIe 内存后端（第 6 步）
- 接口：`alloc / put / get / evict`，支持批量操作和异步执行。
- 存储：一块独立的 pinned 内存池。
- **限速：** 用令牌桶控制带宽和固定延迟，预设 Gen4 x8、Gen5 x8、Gen5 x16 三档，再加一个“共享系数”模拟链路被抢占。
- 接入方式：LMCache 的存储后端，或 vLLM 的 KV Connector。不改 vLLM 内核。
- 模型：Qwen3-8B 的 FP8 或 AWQ 版本，给 KV 留出显存。不要用 27B。

### 步骤 8：真实负载（第 7 步，单卡版）
- 负载：ShareGPT 多轮对话 trace，以及“共享 system prompt / RAG 文档 + 不同问题”的请求。
- 对比组：重算、LMCache CPU 后端（Host DRAM 方案）、模拟 PCIe 内存（三档限速）、LMCache 磁盘后端（NVMe 方案）。
- 扫描参数：复用率 0–90%、会话间隔、上下文长度。
- 指标：TTFT、TPOT、吞吐、P99。
- **关键曲线：** TTFT 随链路带宽的变化（验证 H5），从中得出产品至少需要的 PCIe 规格。
- 注意：32GB 主机内存大约只能给 pinned 池和 CPU 后端留出 16–20GB，这个容量上限要如实记录。

### 步骤 9：预测模型和产品判断（第 8 步）
- 把拷贝模型（步骤 5）、争抢模型（步骤 6）、重叠率（步骤 8）合在一起，预测 8 卡服务器上的 TTFT 和 P99。
- 做敏感性分析：复用率、链路带宽、压缩比，哪个对结论影响最大。
- 输出：一页纸结论，加一份服务器验证清单（5 到 10 个测试点）。

---

## 三、这个环境下无法验证的内容

| 无法验证 | 原因 | 单卡上的替代做法 |
|---|---|---|
| GPU 和第三方设备之间的 P2P DMA | 消费级卡不支持 GPUDirect RDMA 或第三方 P2P | 用 pinned 主机内存模拟，PCIe 路径一致 |
| 同一 Switch 下 P2P 与经 RC 的对比 | 没有 PCIe Switch | 只测经 RC 的路径，得出基线 |
| 多卡争抢上行链路（判断点 1、3） | 只有一张 GPU | 用多进程和排队模型模拟（步骤 6） |
| 1、2、4、8 卡扩展实验 | 只有一张 GPU | 用预测模型外推，留到服务器上验证 |
| ACS 和 IOMMU 对 P2P 的影响 | 没有 Switch，也没有 P2P 路径 | 写好检查脚本（步骤 4），并测 IOMMU 开关对拷贝的影响 |
| Gen5 和 Gen6 的实际带宽 | 4080 是 Gen4 | 用限速模拟三档规格（步骤 7） |
| 真实 PCIe 内存卡的特性（板载 DDR、DMA 引擎、MPS/MRRS） | 没有 FPGA 原型卡 | 留到服务器阶段用 FPGA 测 |
| 70B 规模、TP8、高并发的绝对性能 | 显存和主机内存不够 | 用 8B 模型测出相对规律，再按比例换算 |
| 和 CXL 的对比 | 没有支持 CXL 的平台 | 只在纸面上算账；有条件时用双路服务器的远端 NUMA 模拟 |
| 卡上 KV 压缩的硬件收益 | 没有卡上计算能力 | 用 KV Q8/Q4 量化模拟“压缩比等于带宽倍数”的效果 |

---

## 四、时间安排

| 周 | 内容 | 交付物 |
|---|---|---|
| 1 | 步骤 1–3（Windows） | KV 计算器校准结果、TPOT 拟合、换入与重算对比表 |
| 2 | 装 Linux，做步骤 4–5 | `check_platform.sh`、DMA 特性 CSV、t₀ 和 BW |
| 3–4 | 步骤 7（先不限速，跑通后端） | 模拟 PCIe 内存后端 |
| 5–6 | 加限速，做步骤 8 | TTFT 随复用率、随带宽变化的曲线 |
| 7 | 步骤 6 和步骤 9 | 争抢模型、8 卡预测结果 |
| 8 | 整理 | 一页纸结论、服务器验证清单 |

## 五、继续 / 停止判断点

1. **步骤 1 之后：** 如果模型是混合注意力结构，所有带宽账都要按真实 KV 层数重算后再往下做。
2. **步骤 3 之后：** 如果换入只比重算快不到 10 倍，分层方案在这个模型上收益有限，需要换一个标准结构的模型再测。
3. **步骤 5 之后：** 如果合并后的大块拷贝带宽达不到链路理论值的 80%，先查清原因（是否 pinned、ReBAR、链路代数），再往上层做。
4. **步骤 8 之后：** 如果 TTFT 对带宽不敏感（H5 不成立），P2P 和高规格 PCIe 就不是卖点，产品定位应改为“低成本大容量”。

---

## 附录：工具实现

| 工具 | 对应步骤 | 运行环境 |
|---|---|---|
| A. `kv_calc.py` 算账计算器 | 第 2 步算账、步骤 1、判断点 1 | 任意 Python 3 |
| B. `tpot_fit.py` TPOT 拟合 | 步骤 2 | Python 3 + numpy |
| C. `check_platform.sh` 平台检查 | 步骤 4 | Linux |
| D. `dma_bench.py` 裸 DMA Benchmark | 步骤 5、步骤 6 | Linux + PyTorch(CUDA) |

### A. `kv_calc.py`：KV 容量、带宽和判断点 1

和研究路线页面里的计算器公式一致。另外加了两项：`--kv-layers`，用于混合注意力模型（只有部分层带 KV）；以及“重算 prefill 与换入”的对比。

```python
#!/usr/bin/env python3
"""KV Cache 算账：容量 / decode 带宽 / 换入 vs 重算 / 上行链路占用率（判断点 1）"""
import argparse

def fmt(b):
    for unit, s in ((1e12, "TB"), (1e9, "GB"), (1e6, "MB")):
        if b >= unit:
            return f"{b/unit:.2f} {s}"
    return f"{b/1e3:.0f} KB"

p = argparse.ArgumentParser()
p.add_argument("--layers", type=int, default=80, help="总层数")
p.add_argument("--kv-layers", type=int, default=None, help="带 KV 的层数（混合结构时填写）")
p.add_argument("--kv-heads", type=int, default=8)
p.add_argument("--head-dim", type=int, default=128)
p.add_argument("--bytes", type=float, default=2.0, help="f16=2, q8_0≈1.06, q4_0≈0.56")
p.add_argument("--ctx", type=int, default=32768)
p.add_argument("--conc", type=int, default=64)
p.add_argument("--tp", type=int, default=8)
p.add_argument("--tpot-ms", type=float, default=50)
p.add_argument("--link-gbs", type=float, default=50, help="单链路实际可用带宽")
p.add_argument("--share", type=int, default=2, help="每条上行链路共享的 GPU 数")
p.add_argument("--swapin-rate", type=float, default=4, help="每卡换入请求速率（次/s）")
p.add_argument("--comp", type=float, default=1.0, help="KV 压缩倍数")
p.add_argument("--params-b", type=float, default=70, help="模型参数量（B），用于估算重算")
p.add_argument("--gpu-tflops", type=float, default=990, help="单卡峰值 TFLOPS（H100 FP16 稠密≈990）")
p.add_argument("--mfu", type=float, default=0.4)
a = p.parse_args()

kv_layers = a.kv_layers or a.layers
per_tok = 2 * kv_layers * a.kv_heads * a.head_dim * a.bytes
per_req = per_tok * a.ctx
total = per_req * a.conc
tp = max(1, a.tp)
per_gpu_req = per_req / tp
comp = max(1.0, a.comp)

decode_bw = per_req / (a.tpot_ms / 1e3) / 1e9
swap_ms = per_gpu_req / comp / (a.link_gbs * 1e9) * 1e3
swap_shared_ms = swap_ms * max(1, a.share)
recompute_s = 2 * a.params_b * 1e9 * a.ctx / (tp * a.gpu_tflops * 1e12 * a.mfu)
need_gpu = a.swapin_rate * per_gpu_req / comp / 1e9
need_uplink = need_gpu * max(1, a.share)
util = need_uplink / a.link_gbs * 100

rows = [
    ("每 token KV", fmt(per_tok)),
    ("单请求 KV", f"{fmt(per_req)}（TP{tp} 单卡 {fmt(per_gpu_req)}）"),
    ("总 KV（上下文×并发）", fmt(total)),
    ("decode 读 KV 所需带宽", f"{decode_bw:.0f} GB/s（单请求）"),
    ("单卡换入（独占链路）", f"{swap_ms:.1f} ms"),
    ("单卡换入（共享上行）", f"{swap_shared_ms:.1f} ms"),
    ("重算 prefill 估算", f"{recompute_s*1e3:.0f} ms"),
    ("换入/重算 加速比", f"{recompute_s*1e3/swap_shared_ms:.1f}×（H4 要求 ≥10×）"),
    ("每卡所需换入带宽", f"{need_gpu:.1f} GB/s"),
    ("每条上行链路所需带宽", f"{need_uplink:.1f} GB/s"),
    ("上行链路占用率", f"{util:.0f} %"),
]
for k, v in rows:
    print(f"{k:<16}{v}")

if util >= 70:
    print("判断点 1：继续。Host DRAM 路径上行链路接近饱和，P2P 价值可能成立。")
elif util >= 30:
    print("判断点 1：有条件继续。优势主要在 P99，需在多卡实验重点验证。")
else:
    print("判断点 1：P2P 优势不明显。转向“低成本大容量”定位，或提高目标负载后重算。")
```

用法示例：
```bash
# 70B 服务器目标场景
python kv_calc.py
# 本机 27B（先按步骤 1 查到真实结构再填；以下层数和 head 数为占位值）
python kv_calc.py --layers 64 --kv-layers 16 --kv-heads 4 --head-dim 256 --bytes 0.56 \
  --ctx 32768 --conc 1 --tp 1 --link-gbs 24 --share 1 --params-b 27 --gpu-tflops 195 --mfu 0.3
```

> 阈值 70% / 30% 是经验值，不是实测值。“换入请求速率”对结论影响最大，要用目标场景的真实估算。

### B. `tpot_fit.py`：拟合 TPOT = a + b·L（步骤 2）

输入的 CSV 有两列：`ctx,tpot_ms`，每次重复测量各占一行。

```python
#!/usr/bin/env python3
import sys, csv
import numpy as np

path, kv_bytes_per_tok, pcie_gbs = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
L, T = [], []
with open(path) as f:
    for r in csv.DictReader(f):
        L.append(float(r["ctx"])); T.append(float(r["tpot_ms"]))
L, T = np.array(L), np.array(T)

b, a = np.polyfit(L, T, 1)                      # ms, ms/token
r2 = 1 - ((T - (a + b*L))**2).sum() / ((T - T.mean())**2).sum()
bw_kv = kv_bytes_per_tok / (b / 1e3) / 1e9      # GB/s
print(f"a = {a:.2f} ms（权重读取等固定开销）")
print(f"b = {b*1e3:.4f} ms / 1K tokens,  R² = {r2:.3f}  → H2 {'成立' if r2 > 0.9 and b > 0 else '存疑'}")
print(f"实际读 KV 带宽 ≈ {bw_kv:.0f} GB/s（4080 显存峰值 ~717）")

print("\nctx      实测TPOT   若经PCIe读KV的TPOT")
for l in sorted(set(L)):
    pcie = a + kv_bytes_per_tok * l / (pcie_gbs * 1e9) * 1e3
    print(f"{int(l):<8} {T[L==l].mean():8.1f}   {pcie:10.1f} ms")
```

```bash
python tpot_fit.py tpot.csv <每token KV字节数> <bandwidthTest 实测 GB/s>
```

### C. `check_platform.sh`：平台检查（步骤 4）

```bash
#!/usr/bin/env bash
# 用法：sudo ./check_platform.sh > platform_$(hostname).txt
sec(){ echo; echo "===== $1 ====="; }

sec "GPU 与驱动"
nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv

sec "BAR1（ReBAR 是否开启）"
nvidia-smi -q -d MEMORY | grep -A3 -i "BAR1"

sec "GPU 拓扑（PIX/PXB=同 Switch, PHB/NODE/SYS=经 RC 或跨 Socket）"
nvidia-smi topo -m

sec "PCIe 树"
lspci -tv

sec "每个 NVIDIA 设备的链路状态 / MPS / MRRS"
for d in $(lspci -D -d 10de: | awk '{print $1}'); do
  echo "--- $d"
  lspci -s "$d" -vv 2>/dev/null | grep -E "LnkCap:|LnkSta:|MaxPayload|MaxReadReq"
done

sec "上游桥 / Switch 端口的 ACS 设置（ReqRedir+/CmpltRedir+ 会把 P2P 送回 RC）"
for d in $(lspci -D | awk '/PCI bridge/{print $1}'); do
  acs=$(lspci -s "$d" -vvv 2>/dev/null | grep -i "ACSCtl")
  [ -n "$acs" ] && echo "$d $acs"
done

sec "IOMMU"
cat /proc/cmdline
dmesg 2>/dev/null | grep -iE "iommu|dmar|amd-vi" | head -20
ls /sys/kernel/iommu_groups 2>/dev/null | wc -l | xargs echo "IOMMU groups:"

sec "NUMA"
numactl -H 2>/dev/null || lscpu | grep -i numa

sec "GPU P2P 能力（需 CUDA samples 的 p2pBandwidthLatencyTest，可选）"
command -v p2pBandwidthLatencyTest >/dev/null && p2pBandwidthLatencyTest | head -40 || echo "未安装，跳过"
```

### D. `dma_bench.py`：裸 DMA Benchmark（步骤 5、6）

把 pinned 主机内存当成“PCIe 内存”，测试项和步骤 5 的表格一一对应，结果写成 CSV。

```python
#!/usr/bin/env python3
"""用法：python dma_bench.py --out dma.csv [--tests sweep,merge,dir,streams,overlap]
争抢模拟（步骤 6）：同时启动 N 个  python dma_bench.py --tests sweep --tag pN  然后对比"""
import argparse, csv, time
import torch

p = argparse.ArgumentParser()
p.add_argument("--out", default="dma.csv")
p.add_argument("--tests", default="sweep,merge,dir,streams,overlap")
p.add_argument("--iters", type=int, default=50)
p.add_argument("--tag", default="")
a = p.parse_args()
dev = torch.device("cuda")
rows = []

def rec(test, **kw):
    kw.update(test=test, tag=a.tag); rows.append(kw); print(kw)

def timed(fn, iters):
    """返回每次迭代耗时列表（ms）"""
    for _ in range(3): fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); e.synchronize()
        out.append(s.elapsed_time(e))
    return out

def stats(ts, nbytes):
    ts = sorted(ts)
    med, p99 = ts[len(ts)//2], ts[min(len(ts)-1, int(len(ts)*0.99))]
    return dict(bytes=nbytes, med_ms=round(med, 4), p99_ms=round(p99, 4),
                gbs=round(nbytes / (med / 1e3) / 1e9, 2))

tests = a.tests.split(",")

# 1) block 大小扫描：pinned vs pageable，H2D
if "sweep" in tests:
    for pinned in (True, False):
        for kb in [4, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536]:
            n = kb * 1024
            h = torch.empty(n, dtype=torch.uint8, pin_memory=pinned)
            d = torch.empty(n, dtype=torch.uint8, device=dev)
            ts = timed(lambda: d.copy_(h, non_blocking=pinned), a.iters)
            rec("sweep", pinned=pinned, block_kb=kb, **stats(ts, n))

# 2) 小块逐个拷贝 vs 合并一次：拟合 T = t0 + size/BW
if "merge" in tests:
    nblk, kb = 1000, 32
    n = nblk * kb * 1024
    h = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    d = torch.empty(n, dtype=torch.uint8, device=dev)
    hb, db = h.view(nblk, -1), d.view(nblk, -1)
    def small():
        for i in range(nblk): db[i].copy_(hb[i], non_blocking=True)
    t_small = timed(small, 10)
    t_big = timed(lambda: d.copy_(h, non_blocking=True), 10)
    s_small, s_big = stats(t_small, n), stats(t_big, n)
    t0_us = (s_small["med_ms"] - s_big["med_ms"]) / nblk * 1e3
    rec("merge", mode=f"{nblk}x{kb}KB", **s_small)
    rec("merge", mode="merged", **s_big)
    rec("merge", mode="t0_us_per_copy", t0_us=round(t0_us, 2))

# 3) 方向：H2D / D2H / 双向同时
if "dir" in tests:
    n = 256 * 1024 * 1024
    h1 = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    h2 = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    d1 = torch.empty(n, dtype=torch.uint8, device=dev)
    d2 = torch.empty(n, dtype=torch.uint8, device=dev)
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    rec("dir", mode="H2D", **stats(timed(lambda: d1.copy_(h1, non_blocking=True), 20), n))
    rec("dir", mode="D2H", **stats(timed(lambda: h2.copy_(d2, non_blocking=True), 20), n))
    def bidir():
        cur = torch.cuda.current_stream()
        s1.wait_stream(cur); s2.wait_stream(cur)
        with torch.cuda.stream(s1): d1.copy_(h1, non_blocking=True)
        with torch.cuda.stream(s2): h2.copy_(d2, non_blocking=True)
        cur.wait_stream(s1); cur.wait_stream(s2)
    rec("dir", mode="BIDIR", **stats(timed(bidir, 20), 2 * n))

# 4) 队列深度：1/2/4/8 个 stream 并发 H2D（每块 2MB，共 256MB）
if "streams" in tests:
    blk, nblk = 2 * 1024 * 1024, 128
    h = torch.empty(blk * nblk, dtype=torch.uint8, pin_memory=True).view(nblk, -1)
    d = torch.empty(blk * nblk, dtype=torch.uint8, device=dev).view(nblk, -1)
    for ns in (1, 2, 4, 8):
        ss = [torch.cuda.Stream() for _ in range(ns)]
        def run():
            cur = torch.cuda.current_stream()
            for s in ss: s.wait_stream(cur)
            for i in range(nblk):
                with torch.cuda.stream(ss[i % ns]): d[i].copy_(h[i], non_blocking=True)
            for s in ss: cur.wait_stream(s)
        rec("streams", streams=ns, **stats(timed(run, 20), blk * nblk))

# 5) 拷贝与计算重叠
if "overlap" in tests:
    n = 512 * 1024 * 1024
    h = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    d = torch.empty(n, dtype=torch.uint8, device=dev)
    x = torch.randn(8192, 8192, device=dev, dtype=torch.float16)
    cs, ks = torch.cuda.Stream(), torch.cuda.Stream()
    def compute():
        for _ in range(10): x @ x
    def copy():
        d.copy_(h, non_blocking=True)
    def both():
        cur = torch.cuda.current_stream()
        cs.wait_stream(cur); ks.wait_stream(cur)
        with torch.cuda.stream(cs): copy()
        with torch.cuda.stream(ks): compute()
        cur.wait_stream(cs); cur.wait_stream(ks)
    tc = stats(timed(copy, 10), n)["med_ms"]
    tk = stats(timed(compute, 10), 0)["med_ms"]
    tb = stats(timed(both, 10), n)["med_ms"]
    overlap = (tc + tk - tb) / min(tc, tk) * 100
    rec("overlap", copy_ms=tc, compute_ms=tk, both_ms=tb, overlap_pct=round(overlap, 1))

keys = sorted({k for r in rows for k in r})
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print(f"→ {a.out}")
```

**NVMe 路径（步骤 5 最后一项）** 用 fio 单独测：
```bash
fio --name=kvread --filename=/data/kv.bin --size=8G --rw=read --bs=1M --iodepth=32 \
    --ioengine=io_uring --direct=1 --runtime=30 --time_based --group_reporting
fio --name=kvrand --filename=/data/kv.bin --size=8G --rw=randread --bs=64k --iodepth=32 \
    --ioengine=io_uring --direct=1 --runtime=30 --time_based --group_reporting
```

**争抢模拟（步骤 6）：**
```bash
# N 个进程同时抢 PCIe
for i in 1 2 3 4; do python dma_bench.py --tests sweep --tag p$i --out c$i.csv & done; wait
# DDR 被占满时的 H2D
numactl --cpunodebind=0 ./stream &  python dma_bench.py --tests sweep --tag ddr_busy --out ddr.csv
```

> 说明：在单卡上，`streams` 测试里多个 stream 实际共用 GPU 的 copy engine（4080 有 2 个），所以它测的是“队列深度”的效果，不等于多卡争抢。多卡争抢只能用多进程来近似。
