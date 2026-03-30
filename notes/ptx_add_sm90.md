# PTX ISA — sm_90 (Hopper) 新增指令全览

> 基于 PTX ISA 9.1 本地文档搜索整理。
> 涵盖 PTX 7.8 → 8.2 中为 `sm_90` / `sm_90a` 引入的所有新增或扩展指令。
> `sm_90a` 表示该指令仅在 Hopper 架构专属模式（architecture-specific）下可用。

---

## 目录

1. [Warpgroup 级矩阵乘累加 (WGMMA)](#1-warpgroup-级矩阵乘累加-wgmma)
2. [Tensor Memory Accelerator (TMA) — 批量异步拷贝](#2-tensor-memory-accelerator-tma--批量异步拷贝)
3. [Cluster 同步与通信](#3-cluster-同步与通信)
4. [Multimem / NVLink Multicast 内存操作](#4-multimem--nvlink-multicast-内存操作)
5. [异步 Store 与 Reduction](#5-异步-store-与-reduction)
6. [Grid 执行控制](#6-grid-执行控制)
7. [Warp/Thread 管理](#7-warpthread-管理)
8. [Warp 级矩阵操作扩展](#8-warp-级矩阵操作扩展)
9. [数据类型扩展 (BF16 / FP8 / 打包整数)](#9-数据类型扩展-bf16--fp8--打包整数)
10. [内存作用域扩展 (.cluster scope)](#10-内存作用域扩展-cluster-scope)
11. [新增特殊寄存器](#11-新增特殊寄存器)
12. [新增 Cluster 指令集指示符](#12-新增-cluster-指令集指示符)

---

## 1. Warpgroup 级矩阵乘累加 (WGMMA)

**最低要求：`sm_90a`（PTX 8.0 引入，PTX 8.2 扩展稀疏变体）**
**文档位置：** `ptx-docs/9-instruction-set/9.7.15.*`

WGMMA 是 Hopper 最核心的新特性，以 warpgroup（4 个 warp = 128 线程）为单位执行异步矩阵乘累加，比 Ampere 的 `mma` 吞吐量提升约 2×。

| 指令 | 说明 | PTX 版本 |
|------|------|---------|
| `wgmma.mma_async` | 异步 warpgroup 级 MMA，支持 f16/bf16/f32/s8/u8/tf32 等多种类型 | 8.0 |
| `wgmma.mma_async.sp` | 稀疏矩阵 A 的异步 MMA（A 矩阵 2:4 稀疏） | 8.2 |
| `wgmma.fence` | 建立 wgmma 操作与寄存器读写之间的顺序语义 | 8.0 |
| `wgmma.commit_group` | 提交当前所有待提交的 wgmma 操作为一个 group | 8.0 |
| `wgmma.wait_group` | 等待直到 pending wgmma group 数量 ≤ N | 8.0 |

**典型语法：**
```ptx
// D = A * B + D  (f16 输入，f32 累加，m64n256k16)
wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16
    {d0,...,d127}, [descA], [descB], 1, 1, 1, 0, 0;

wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 0;  // 等待所有完成
```

---

## 2. Tensor Memory Accelerator (TMA) — 批量异步拷贝

**最低要求：`sm_90`（PTX 8.0 引入）**
**文档位置：** `ptx-docs/9-instruction-set/9.7.9.25-28`

TMA 通过硬件 tensormap 描述符，将多维张量数据直接从全局内存异步搬运到共享内存（或反向），支持 1D-5D 张量、im2col、swizzle 等。

### 2.1 批量拷贝

| 指令 | 方向 | 说明 |
|------|------|------|
| `cp.async.bulk.shared::cta.global` | global → shared | 普通批量拷贝，mbarrier 完成通知 |
| `cp.async.bulk.shared::cluster.global` | global → cluster shared | 集群共享内存拷贝，支持 `.multicast::cluster`（`sm_90a` 最优） |
| `cp.async.bulk.global.shared::cta` | shared → global | 反向拷贝，bulk_group 完成机制 |
| `cp.async.bulk.shared::cluster.shared::cta` | cta shared → cluster shared | CTA 间共享内存拷贝 |

**语法示例：**
```ptx
cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes
    [dstMem], [srcMem], size, [mbar];
```

### 2.2 TMA Tensor 描述符拷贝

| 指令 | 说明 |
|------|------|
| `cp.async.bulk.tensor.1d` ~ `5d` `.shared::cta.global` | 使用 tensormap 描述符，按张量维度拷贝 |
| `cp.async.bulk.tensor.1d` ~ `5d` `.global.shared::cta` | 反向（store via TMA） |
| `cp.async.bulk.tensor.*d` `.shared::cta.global.im2col` | im2col 模式（卷积用） |
| `cp.async.bulk.prefetch.tensor.*d.L2.global` | 将张量数据预取到 L2 |
| `cp.async.bulk.prefetch.L2.global` | 通用批量 L2 预取 |

### 2.3 Tensormap 运行时修改

| 指令 | 说明 |
|------|------|
| `tensormap.replace` | 运行时修改 tensormap 描述符字段（维度、步长、swizzle 模式等） |
| `prefetch.tensormap` | 将 tensormap 描述符预取到 L1 缓存 |
| `cp_fenceproxy` | tensormap 修改后的 fence，确保修改对后续 TMA 操作可见 |

**`tensormap.replace` 语法：**
```ptx
tensormap.replace.tile.global_address.shared::cta.b1024.b64
    [tm], new_addr;
```

---

## 3. Cluster 同步与通信

**最低要求：`sm_90`（PTX 7.8 引入）**
**文档位置：** `ptx-docs/9-instruction-set/9.7.13.*`

Hopper 引入 CTA Cluster 概念，多个 CTA 可协同执行，共享 L2 可见的 shared memory。

### 3.1 新增指令

| 指令 | 说明 |
|------|------|
| `barrier.cluster.arrive` | 线程到达 cluster 屏障（不等待） |
| `barrier.cluster.wait` | 等待整个 cluster 所有 CTA 到达屏障 |
| `barrier.cluster.arrive.aligned` | 全 warp 对齐版本 |
| `mapa` | 将本 CTA 的 shared 地址映射到 cluster 内指定 CTA 的对应地址 |
| `getctarank` | 查询包含给定地址的 CTA 在 cluster 内的 rank |

**语法：**
```ptx
barrier.cluster.arrive;
barrier.cluster.wait;

// 映射到 cluster 中 rank=1 的 CTA 的共享内存地址
mapa.shared::cluster.u32 mapped_addr, local_addr, 1;

// 获取包含地址 addr 的 CTA rank
getctarank.shared::cluster.u32 rank, addr;
```

### 3.2 mbarrier 扩展（sm_90 新增部分）

| 扩展 | 说明 |
|------|------|
| `mbarrier.arrive` 带 `.cluster` scope | Cluster 内任意 CTA 触发到达 |
| `mbarrier.arrive` 带 `.expect_tx` | 声明将要完成的 transaction byte 数 |
| `mbarrier.complete_tx` | 手动标记已完成若干字节的异步事务 |
| `mbarrier.try_wait` | 非阻塞相位检查（PTX 7.8，`sm_90+`） |
| `mbarrier.try_wait` 带 `.cluster` scope | Cluster 可见版本 |

```ptx
// 初始化 mbarrier，期望 128B 的异步事务完成
mbarrier.init.shared.b64 [mbar], 1;
mbarrier.arrive.expect_tx.shared::cta.b64 _, [mbar], 128;

// 等待完成（循环 try_wait）
mbarrier.try_wait.shared::cta.b64 complete, [mbar], phase;
```

---

## 4. Multimem / NVLink Multicast 内存操作

**最低要求：`sm_90`（PTX 8.1 引入）**
**文档位置：** `ptx-docs/9-instruction-set/9.7.9.14, 9.7.9.26, 9.7.9.27`

Multimem 地址允许单条指令同时对多个 NVLink 连接的 GPU 上的内存执行操作（broadcast / reduce）。

| 指令 | 说明 |
|------|------|
| `multimem.ld_reduce.global.{op}.{type}` | 从 multimem 地址 load，执行 reduction（如 `.add.f32`） |
| `multimem.st.global.{type}` | 向 multimem 地址广播 store |
| `multimem.red.global.{op}.{type}` | 向 multimem 地址执行原子 reduction |
| `multimem.cp_async.bulk` | 批量异步拷贝到 multimem 地址（含 multicast） |
| `multimem.cp_reduce_async.bulk` | 批量异步 reduce 到 multimem 地址 |

```ptx
multimem.ld_reduce.relaxed.cluster.global.add.f32 %f1, [addr];
multimem.st.relaxed.cluster.global.f32 [addr], %f1;
multimem.red.relaxed.cluster.global.add.f32 [addr], %f1;
```

---

## 5. 异步 Store 与 Reduction

**最低要求：`sm_90`（PTX 8.1 引入）**
**文档位置：** `ptx-docs/9-instruction-set/9.7.9.12, 9.7.13.7`

| 指令 | 说明 |
|------|------|
| `st.async.shared.{type}` | 异步将寄存器值 store 到共享内存，mbarrier 通知完成 |
| `red.async.shared.{op}.{type}` | 异步对共享内存地址执行 reduction，mbarrier 通知完成 |

```ptx
st.async.shared::cta.mbarrier::complete_tx::bytes.b32 [dst], val, [mbar];
red.async.relaxed.cta.shared::cta.mbarrier::complete_tx::bytes.add.f32
    [dst], val, [mbar];
```

---

## 6. Grid 执行控制

**最低要求：`sm_90`（PTX 7.8 引入）**
**文档位置：** `ptx-docs/9-instruction-set/9.7.13.13`

| 指令 | 说明 |
|------|------|
| `griddepcontrol.launch_dependents` | 通知 GPU 调度器可以提前启动依赖当前 grid 的后续 grid |
| `griddepcontrol.wait` | 等待所有依赖前驱 grid 完成后再继续 |

这是 CUDA Graph 细粒度流水线的 PTX 基础。

```ptx
griddepcontrol.launch_dependents;
// ... 计算部分完成 ...
griddepcontrol.wait;
```

---

## 7. Warp/Thread 管理

**最低要求：`sm_90a`（PTX 8.0 引入）**
**文档位置：** `ptx-docs/9-instruction-set/9.7.13.14, 9.7.19.5`

| 指令 | 说明 |
|------|------|
| `elect.sync.b32` | warp 内选举一个线程（返回 predicate），用于 WGMMA leader thread 模式 |
| `setmaxnreg.inc.sync.aligned.u32` | 动态增加当前 warp 的最大寄存器数（为 WGMMA 腾出寄存器）|
| `setmaxnreg.dec.sync.aligned.u32` | 动态减少当前 warp 的最大寄存器数 |

```ptx
// 选举 warp 内 leader
elect.sync.b32 is_leader_pred|elected_lane, membermask;

// WGMMA consumer warp 增加寄存器
setmaxnreg.inc.sync.aligned.u32 232;
// Producer warp 减少寄存器
setmaxnreg.dec.sync.aligned.u32 40;
```

---

## 8. Warp 级矩阵操作扩展

**文档位置：** `ptx-docs/9-instruction-set/9.7.14.5`

| 指令 | 最低要求 | 说明 |
|------|---------|------|
| `stmatrix.sync.aligned` | `sm_90` | 将寄存器中的矩阵片段 store 到 shared memory（与 `ldmatrix` 对称） |
| `mma.sync.f64` `.m16n8k4/k8/k16` | `sm_90` | f64 类型 MMA 支持更多 shape |

```ptx
// 存储 2 个 8x8 f16 矩阵到 shared memory
stmatrix.sync.aligned.x2.m8n8.shared::cta.b16 [addr], {r0, r1};
```

---

## 9. 数据类型扩展 (BF16 / FP8 / 打包整数)

**最低要求：`sm_90`（PTX 7.8 引入，PTX 8.0 扩展打包整数）**

### 9.1 BF16 算术指令

以下指令均扩展支持 `.bf16` / `.bf16x2` 类型：

| 指令 | 新增类型 |
|------|---------|
| `add` | `.bf16`, `.bf16x2` |
| `sub` | `.bf16`, `.bf16x2` |
| `mul` | `.bf16`, `.bf16x2` |
| `fma` | `.f16`, `.f16x2`, `.bf16`, `.bf16x2`（含 `.oob` 修饰符） |
| `tanh.approx` | `.bf16`, `.bf16x2` |
| `ex2.approx` | `.bf16`, `.bf16x2` |
| `set` | 输入/输出支持 `.bf16`, `.bf16x2` |
| `setp` | `.bf16`, `.bf16x2` |

### 9.2 FP8 转换

| 指令 | 新增支持 |
|------|---------|
| `cvt` | `.e4m3x2` / `.e5m2x2` ↔ `f32`, `f16x2`（FP8 打包格式） |
| `cvt` | `.bf16` ↔ 整数/f16/f64 互转 |
| `cvt.tf32.f32.relu` | TF32 with ReLU 融合 |
| `atom` / `red` | `.bf16`, `.e4m3`, `.e5m2` 类型 |

### 9.3 打包整数（PTX 8.0）

| 指令 | 新增类型 |
|------|---------|
| `add` | `.u16x2`, `.s16x2` |
| `min` | `.u16x2`, `.s16x2`, `.relu.s32` |
| `max` | `.u16x2`, `.s16x2`, `.relu.s32` |

---

## 10. 内存作用域扩展 (.cluster scope)

**最低要求：`sm_90`（PTX 7.8 引入）**

Hopper 新增 `.cluster` 内存作用域，使得 cluster 内所有 CTA 对 shared memory 有统一可见性。

| 指令 | 扩展内容 |
|------|---------|
| `ld` | 支持 `.cluster` scope；支持 `.shared::cluster` 地址空间 |
| `st` | 支持 `.cluster` scope；支持 `.shared::cluster` 地址空间 |
| `atom` | 支持 `.cluster` scope |
| `red` | 支持 `.cluster` scope |
| `fence` / `membar` | 支持 `.cluster` scope |
| `isspacep` | 支持 `::cluster` 子限定符检查 |
| `cvta` | 支持 `::cluster` 子限定符地址转换 |

```ptx
// 从 cluster 内另一 CTA 的 shared memory 读取
ld.shared::cluster.u32 %r0, [mapped_addr];

// cluster 级 fence
fence.cluster.sc;
membar.cluster;
```

---

## 11. 新增特殊寄存器

**最低要求：`sm_90`（PTX 7.8 / 8.0 / 8.1 引入）**
**文档位置：** `ptx-docs/10-special-registers/`

### Cluster 信息寄存器（PTX 7.8）

| 寄存器 | 类型 | 说明 |
|--------|------|------|
| `%is_explicit_cluster` | `.u32` | 当前 kernel 是否使用 explicit cluster |
| `%clusterid.{x,y,z}` | `.u32` | 当前 cluster 在 grid 中的 ID |
| `%nclusterid.{x,y,z}` | `.u32` | grid 中 cluster 的总数 |
| `%cluster_ctaid.{x,y,z}` | `.u32` | 当前 CTA 在 cluster 中的 ID |
| `%cluster_nctaid.{x,y,z}` | `.u32` | cluster 中 CTA 的总数 |
| `%cluster_ctarank` | `.u32` | 当前 CTA 在 cluster 中的线性 rank |
| `%cluster_nctarank` | `.u32` | cluster 中 CTA 总数（线性） |

### 其他新增寄存器

| 寄存器 | PTX 版本 | 说明 |
|--------|---------|------|
| `%current_graph_exec` | 8.0 | 标识当前 CUDA device graph 的执行句柄 |
| `%aggr_smem_size` | 8.1 | 当前 CTA 的聚合 shared memory 大小（含动态分配） |

---

## 12. 新增 Cluster 指令集指示符

**最低要求：`sm_90`（PTX 7.8 引入）**
**文档位置：** `ptx-docs/11-directives/`

| 指示符 | 说明 |
|--------|------|
| `.reqnctapercluster nx, ny, nz` | 声明 kernel 所需的每 cluster CTA 数量 |
| `.explicitcluster` | 声明该 kernel 使用 explicit cluster 模式 |
| `.maxclusterrank N` | 声明 cluster 的最大 rank 数上限 |

```ptx
.entry myKernel(.param .u64 param0)
.reqnctapercluster 2, 1, 1
.explicitcluster
{
    ...
}
```

---

## 快速搜索参考

```bash
# 搜索所有 sm_90 相关指令文件
grep -rl "sm_90" .claude/skills/cuda/references/ptx-docs/9-instruction-set/

# 查看 WGMMA 指令完整语法
cat .claude/skills/cuda/references/ptx-docs/9-instruction-set/9.7.15.5-*.md | grep -A5 "Syntax"

# 查看 TMA 批量拷贝
grep -n "cp.async.bulk" .claude/skills/cuda/references/ptx-docs/9-instruction-set/9.7.9.25-*.md | head -30

# 查看 Cluster 特殊寄存器
ls .claude/skills/cuda/references/ptx-docs/10-special-registers/ | grep cluster
```

---

## 参考来源

| 文档章节 | 内容 |
|---------|------|
| Release Notes 13.12 (PTX 7.8) | sm_90 首次引入：cluster、mbarrier、griddepcontrol、BF16/FP8 |
| Release Notes 13.11 (PTX 8.0) | sm_90a：WGMMA、bulk copy、elect.sync、setmaxnreg、packed int |
| Release Notes 13.10 (PTX 8.1) | multimem、st.async、red.async、%aggr_smem_size |
| Release Notes 13.9 (PTX 8.2) | wgmma.mma_async.sp（稀疏 WGMMA） |
| PTX ISA 9.1 本地文档 | `ptx-docs/9-instruction-set/`、`ptx-docs/10-special-registers/` |
