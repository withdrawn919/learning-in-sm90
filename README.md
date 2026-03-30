# hopper teaching&learning
你是精通hopper的资深工程师，且熟悉使用nvidia的cutlass/cute编写hopper上运行的程序。参考nvidia的官方文档以及知乎文档等以及相关的git仓库，给出一个初学者学习hopper新特性的step by step计划，每一步都只包含一个新特性，并包含一个使用CUTE编写的最小的能执行的代码。

需要你将将每次生成的计划按照markdown格式写到notes文件夹中，同时将参考文献的地址和git仓库的地址记录到README.md中。

---

## 参考文章与文档

### Step 0: CuTe 基础 — Layout 与 Tensor
- [CUTLASS CuTe 官方文档](https://docs.nvidia.com/cutlass/latest/overview.html) — CuTe 的 Layout、Tensor、TiledCopy 等核心概念的权威文档
- [CUTLASS CuTe sgemm_1 教程代码](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_1.cu) — 最基础的 CuTe GEMM 示例，展示 make_tensor / local_tile / local_partition 的用法

### Step 1: TMA Load — 全局内存到共享内存的异步拷贝
- [Colfax Research: Mastering the NVIDIA Tensor Memory Accelerator (TMA)](https://research.colfax-intl.com/tutorial-hopper-tma/) — 详解 TMA Load/Store/Multicast 的原理和 CUTLASS API 用法
- [知乎: 【教程翻译】掌握 NVIDIA 张量内存加速器 (TMA)](https://zhuanlan.zhihu.com/p/721901629) — 上述 Colfax 文章的中文翻译
- [CUTLASS wgmma_tma_sm90.cu 教程代码](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/hopper/wgmma_tma_sm90.cu) — 完整的 TMA + WGMMA GEMM 示例，展示 make_tma_atom / tma_partition / SM90_TMA_LOAD
- [PyTorch Blog: Deep Dive on the Hopper TMA Unit for FP8 GEMMs](https://pytorch.org/blog/hopper-tma-unit/) — 从 PyTorch 视角解析 TMA 在 FP8 GEMM 中的应用

### Step 2: TMA Store — 共享内存到全局内存的异步写回
- [Colfax Research: Mastering the NVIDIA Tensor Memory Accelerator (TMA)](https://research.colfax-intl.com/tutorial-hopper-tma/) — 文中包含 TMA Store 和 TMA Store Reduce 的详细说明
- [CUTLASS copy_sm90_tma.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp) — SM90_TMA_STORE 的实现源码，包含 tma_store_arrive / tma_store_wait 接口

### Step 3: Swizzle — 共享内存 Bank Conflict 消除
- [Lei Mao's Log Book: CuTe Swizzle](https://leimao.github.io/blog/CuTe-Swizzle/) — 系统讲解 CuTe Swizzle<B,M,S> 的数学原理和参数含义
- [Understanding CuTe Swizzling - The Math Behind 32B, 64B, and 128B Patterns](https://veitner.bearblog.dev/understanding-cute-swizzling-the-math-behind-32b-64b-and-128b-patterns/) — 深入解析 Swizzle<0/1/2/3, 4, 3> 对应的 NoSwizzle/32B/64B/128B 模式
- [Swizzles and their usage in CuTeDSL Kernels](https://veitner.bearblog.dev/swizzles-and-their-usage-in-cutedsl-kernels/) — Swizzle 在 CuTeDSL 中的实际使用方式
- [CUTLASS Discussion #1018: How to understand Swizzle in CuTe](https://github.com/NVIDIA/cutlass/issues/1018) — 社区讨论，包含 NVIDIA 工程师的回复
- [CUTLASS swizzle.hpp 源码](https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle.hpp) — Swizzle 的 XOR 实现和注释

### Step 4: 异步事务屏障（Async Transaction Barrier）
- [Colfax Research: Mastering the NVIDIA TMA](https://research.colfax-intl.com/tutorial-hopper-tma/) — 详解 ClusterTransactionBarrier 与 TMA 的配合：arrive_and_expect_tx / wait
- [CUTLASS wgmma_tma_sm90.cu](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/hopper/wgmma_tma_sm90.cu) — 展示 ProducerBarType / ConsumerBarType 的初始化和同步流程

### Step 5: Cluster 与分布式共享内存（DSMEM）
- [NVIDIA Blog: Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) — 介绍 Thread Block Cluster、Distributed Shared Memory 的架构设计
- [NVIDIA CUDA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) — Cluster 的调优建议和最大 cluster size
- [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture (论文)](https://arxiv.org/pdf/2402.13499) — 实测 DSMEM 比全局内存快约 7 倍
- [Colfax Research: GEMM with Thread Block Clusters](https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/) — Cluster 在 GEMM 中的应用

### Step 6: WGMMA（Warp Group Matrix Multiply-Accumulate）
- [Colfax Research: Fast Matrix-Multiplication with WGMMA on Hopper GPUs](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) — 最详细的 WGMMA 教程，讲解 SM90_64x64x16 指令、SS/RS 模式、描述符生成
- [CUTLASS wgmma_sm90.cu 教程代码](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/hopper/wgmma_sm90.cu) — 纯 WGMMA 示例（不含 TMA），展示 warpgroup_arrive / commit_batch / wait 调用序列
- [DeepWiki: SM90 Hopper Architecture Features](https://deepwiki.com/NVIDIA/cutlass/7.1-sm90-hopper-architecture) — CUTLASS 中 SM90 MMA Atom 的命名规则和用法

### Step 7: Pipeline 与多阶段异步流水线
- [Colfax Research: Efficient GEMM kernel designs with Pipelining](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/) — 从 0-stage 到 3-stage 流水线，详解 PipelineState、生产者-消费者同步
- [CUTLASS wgmma_tma_sm90.cu](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/hopper/wgmma_tma_sm90.cu) — 3-stage pipeline 的完整参考实现
- [CUTLASS sm90_pipeline.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/pipeline/sm90_pipeline.hpp) — PipelineTmaAsync 等流水线类的定义

### Step 8: Warp Specialization（Warp 特化）
- [Colfax Research: Efficient GEMM kernel designs with Pipelining](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/) — 文中第三节介绍 warp specialization 和 ping-pong scheduling
- [PyTorch Blog: Deep Dive on CUTLASS Ping-Pong GEMM Kernel](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/) — 解析 Ping-Pong 双缓冲 warp 特化策略，Producer 减少寄存器以提高 occupancy
- [CUTLASS sm90_gemm_tma_warpspecialized_pingpong.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp) — Ping-Pong warp 特化内核的生产级实现

### Step 9: 持久化内核（Persistent Kernel）与 Tile Scheduler
- [Colfax Research: Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/) — 持久化内核的动机、Tile Scheduler 的调度策略、Stream-K 负载均衡
- [CUTLASS sm90_tile_scheduler_stream_k.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp) — Stream-K tile 调度器源码
- [Colfax Research: Developing CUDA Kernels for GEMM on Hopper (PDF)](https://research.colfax-intl.com/wp-content/uploads/2023/12/colfax-gemm-kernels-hopper.pdf) — 学术论文，包含持久化内核的性能分析

### Step 10: FP8 数据类型支持
- [CUTLASS example 54: Hopper FP8 Warp Specialized GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/54_hopper_fp8_warp_specialized_gemm) — FP8 E4M3/E5M2 WGMMA GEMM 的完整示例
- [PyTorch Blog: Deep Dive on the Hopper TMA Unit for FP8 GEMMs](https://pytorch.org/blog/hopper-tma-unit/) — FP8 场景下 TMA 的优化策略
- [CUTLASS example 55: Hopper Mixed Dtype GEMM](https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_fp8_gemm.cu) — INT4 x FP8 混合精度 GEMM

### Step 11: 综合实战 — 基于 CuTe 的高性能 GEMM
- [CUTLASS wgmma_tma_sm90.cu](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/hopper/wgmma_tma_sm90.cu) — 整合 TMA + WGMMA + Pipeline + Swizzle 的完整可运行示例
- [CUTLASS example 48: Hopper Warp Specialized GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/48_hopper_warp_specialized_gemm) — 生产级 Hopper GEMM，使用 CollectiveBuilder 高级 API
- [Colfax Research: Developing CUDA Kernels for GEMM on Hopper](https://research.colfax-intl.com/nvidia-hopper-gemm-cutlass/) — 端到端 Hopper GEMM 的开发指南
- [FlashAttention-2 on Hopper (论文)](https://arxiv.org/html/2312.11918v1) — 实战案例：在 Hopper 上使用 CUTLASS 实现 FlashAttention-2

---

## 参考 Git 仓库
- [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) — NVIDIA 官方 CUTLASS 库，包含 CuTe、所有 Hopper 示例和教程
- [reed-lau/cute-gemm](https://github.com/reed-lau/cute-gemm) — 基于 CuTe 的 GEMM 实现示例
- [BBuf/how-to-optim-algorithm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda) — CUDA 优化算法合集，包含 CuTe/CUTLASS 学习资料

---

## 参考文章汇总（按来源）

### Colfax Research 系列教程
- [WGMMA 教程](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) — WGMMA 指令详解与实战
- [TMA 教程](https://research.colfax-intl.com/tutorial-hopper-tma/) — TMA Load/Store/Multicast 全面讲解
- [Pipeline 教程](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/) — 流水线设计与 Warp Specialization
- [Persistent Kernel 教程](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/) — 持久化内核与 Stream-K

### 知乎中文教程
- [TMA 教程翻译](https://zhuanlan.zhihu.com/p/721901629) — Colfax TMA 教程的中文翻译
- [知乎 - 滕德诚](https://www.zhihu.com/people/teng-de-cheng-32/posts) — CuTe/CUTLASS 相关博文
- [知乎 - 朱熹嘉初](https://www.zhihu.com/people/zhu-xi-jia-chu-38/posts) — CUDA/GPU 优化相关
- [知乎专栏 - CUDA 编程](https://www.zhihu.com/column/c_1696937812497235968) — CUDA 编程专栏

### 博客文章
- [Lei Mao: CuTe Swizzle](https://leimao.github.io/blog/CuTe-Swizzle/) — Swizzle 数学原理
- [Simon's Blog: Understanding CuTe Swizzling](https://veitner.bearblog.dev/understanding-cute-swizzling-the-math-behind-32b-64b-and-128b-patterns/) — Swizzle 模式详解
- [PyTorch Blog: CUTLASS Ping-Pong GEMM](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/) — Warp Specialization 深度解析
- [Kapil Sharma: Learn CUTLASS the hard way](https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way-2/) — CUTLASS 实战入门
