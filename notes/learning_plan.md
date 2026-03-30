# Hopper 架构 + CuTe 初学者学习计划

> 每一步聚焦一个 Hopper 新特性，配合一个使用 CuTe 编写的最小可执行代码示例。

---

## Step 0: 环境准备与 CuTe 基础
- CUTLASS 编译环境搭建（CUDA 12+, sm_90）
- CuTe 核心概念：Layout、Tensor、TiledCopy、TiledMMA

## Step 1: TMA（Tensor Memory Accelerator）— 全局内存到共享内存的异步拷贝
- TMA 描述符（TmaDescriptor）创建
- 使用 `cute::SM90_TMA_LOAD` 进行 1D/2D Tensor 加载

## Step 2: TMA Store — 共享内存到全局内存的异步写回
- 使用 `cute::SM90_TMA_STORE` 将结果写回全局内存
- TMA Store 与 TMA Load 的配合

## Step 3: 异步事务屏障（Async Transaction Barrier / TMA Multicast）
- `cute::SM90_TMA_LOAD_MULTICAST` 多播加载
- 异步屏障（`arrive` / `wait`）机制

## Step 4: Cluster 与分布式共享内存（DSMEM）
- Thread Block Cluster 概念
- 跨 SM 的分布式共享内存读写

## Step 5: WGMMA（Warp Group Matrix Multiply-Accumulate）
- Warp Group 概念（4 个 Warp 协作）
- 使用 `cute::SM90_64x128x16_F16F16F16_SS` 等 WGMMA 指令

## Step 6: Pipeline 与多阶段异步流水线
- 生产者-消费者模型
- 多阶段（multi-stage）流水线：TMA Load → WGMMA Compute 重叠

## Step 7: Warp Specialization（Warp 特化）
- 生产者 Warp 与消费者 Warp 的分工
- `cute::ProducerWarp` / `cute::ConsumerWarp` 编程模式

## Step 8: 持久化内核（Persistent Kernel）与 Tile Scheduler
- 持久化内核的动机与实现
- Tile Scheduler 的工作窃取（work stealing）策略

## Step 9: FP8 数据类型支持
- Hopper 原生 FP8（E4M3 / E5M2）支持
- 使用 WGMMA 执行 FP8 矩阵乘法

## Step 10: 综合实战 — 基于 CuTe 的高性能 GEMM
- 整合以上特性实现一个完整的 Hopper GEMM Kernel
- 性能对比与调优要点

---

## 参考资料
见 [README.md](../README.md) 中的参考文章与 Git 仓库列表。
