# Hopper 架构 + CuTe 初学者学习计划 - GPT 5.4
下面这条路线，我按“**每一步只引入一个 Hopper 新特性**”来排，目标不是一上来就写完整 GEMM，而是先把 Hopper 的几个核心抽象一个个吃透：**cluster → DSM → mbarrier → TMA 的坐标/描述符思维 → WGMMA → warp specialization**。Hopper 上的线程块簇、分布式共享内存和 TMA 都是从 compute capability 9.0 开始支持；CUTLASS/CuTe 的 Hopper 路径通常要求 `sm_90a`/`90a` 编译目标，官方 quickstart 推荐 C++17，Hopper 示例要求 CUDA 12.0+。([NVIDIA Docs][1])

先说结论：**最适合初学者的学习顺序不是“先 GEMM”，而是“先同步与数据搬运，再算子，再整合成管线”**。官方文档把 Hopper 的高性能 GEMM 组织成 producer warp group 用 TMA 搬运、consumer warp group 用 WGMMA 计算的结构；CUTLASS 3.x/4.x 又是用 CuTe 把这些层次拼起来。所以你先学清楚底层协作语义，后面的 CUTLASS/CuTe 模板就不会像“天书”。([NVIDIA Docs][2])

---

## 预备步：先把环境固定住

建议直接用：

* CUDA 12.5+ 或更新的 12.x
* CUTLASS 主分支或 4.4.x
* `-std=c++17`
* `-arch=sm_90a`
* Linux + GCC 9/10/11

CUTLASS quickstart 给出的 Hopper 编译方式就是 `-DCUTLASS_NVCC_ARCHS=90a`；官方也明确写了 Hopper 示例要求 compute capability 90/90a 且 CUDA 12.0+。([NVIDIA Docs][3])

最小编译命令模板：

```bash
nvcc -std=c++17 -arch=sm_90a \
  -I/path/to/cutlass/include \
  -I/path/to/cutlass/tools/util/include \
  your_file.cu -o your_file
```

---

# Step 1：只学一个新特性 —— Thread Block Cluster

这一阶段只理解一件事：**Hopper 新增了 cluster 这一层执行层级**。先别碰 DSM、TMA、WGMMA。你只需要知道：多个 CTA 可以组成一个 cluster，彼此在硬件上被看成一个更紧的协作单元。CUDA Programming Guide 明确说明了 thread block clusters 是 CC 9.0+ 的特性。([NVIDIA Docs][1])

### 你要学会什么

1. `__cluster_dims__`
2. `cooperative_groups::this_cluster()`
3. `cluster.block_rank()`

### 最小可执行代码

这个程序只做一件事：启动一个 `2 x 1 x 1` 的 cluster，然后把每个 block 在 cluster 内的 rank 写回全局内存。代码里故意用了 CuTe 的 `Tensor` 包一下全局内存，先建立“用 CuTe 看数据”的习惯。

```cpp
// step1_cluster.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cute/tensor.hpp>

namespace cg = cooperative_groups;
using namespace cute;

__global__ __cluster_dims__(2, 1, 1)
void cluster_rank_kernel(int* out) {
  auto cluster = cg::this_cluster();

  Tensor gOut = make_tensor(make_gmem_ptr(out), make_shape(Int<2>{}));

  if (threadIdx.x == 0) {
    gOut(cluster.block_rank()) = int(cluster.block_rank());
  }
}

int main() {
  int* d_out;
  int h_out[2] = {-1, -1};

  cudaMalloc(&d_out, 2 * sizeof(int));
  cudaMemset(d_out, 0xff, 2 * sizeof(int));

  cluster_rank_kernel<<<2, 128>>>(d_out);
  cudaDeviceSynchronize();

  cudaMemcpy(h_out, d_out, 2 * sizeof(int), cudaMemcpyDeviceToHost);

  printf("cluster ranks = [%d, %d]\n", h_out[0], h_out[1]);

  cudaFree(d_out);
  return 0;
}
```

编译：

```bash
nvcc -std=c++17 -arch=sm_90a \
  -I/path/to/cutlass/include \
  step1_cluster.cu -o step1_cluster
```

预期输出：

```text
cluster ranks = [0, 1]
```

### 通过标准

你能清楚解释：

* grid 里有 2 个 block
* 这 2 个 block 不是“普通并列”，而是一个 cluster
* `block_rank()` 是 cluster 内编号，不是 `blockIdx.x`

---

# Step 2：只学一个新特性 —— Distributed Shared Memory

这一阶段只加一个概念：**cluster 内可以访问彼此的 shared memory**。这是 Hopper 很关键的能力，CUDA guide 也明确说 distributed shared memory 依赖 thread block cluster，且访问远端 shared memory 前后需要 `cluster.sync()` 保证所有 block 都存在且不会提前退出。([NVIDIA Docs][1])

### 你要学会什么

1. `cluster.map_shared_rank`
2. 为什么要 `cluster.sync()`
3. “shared memory 仍然是 per-block 分配，但 cluster 可形成 DSM 地址空间”

### 最小可执行代码

```cpp
// step2_dsm.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cute/tensor.hpp>

namespace cg = cooperative_groups;
using namespace cute;

__global__ __cluster_dims__(2, 1, 1)
void dsm_sum_kernel(int* out) {
  extern __shared__ int smem[];
  auto cluster = cg::this_cluster();

  // 每个 block 在自己的 shared memory 写一个值
  if (threadIdx.x == 0) {
    smem[0] = cluster.block_rank() + 1;   // block0 写 1，block1 写 2
  }

  cluster.sync();  // 保证两个 block 都已写完

  // 只让 rank 0 读取整个 cluster 的 DSM
  if (cluster.block_rank() == 0 && threadIdx.x == 0) {
    int sum = 0;
    for (int r = 0; r < cluster.dim_blocks().x; ++r) {
      int* remote = cluster.map_shared_rank(smem, r);
      sum += remote[0];
    }
    Tensor gOut = make_tensor(make_gmem_ptr(out), make_shape(Int<1>{}));
    gOut(0) = sum;   // 1 + 2 = 3
  }

  cluster.sync();  // 防止别的 block 还在读 DSM 时提前退出
}

int main() {
  int* d_out;
  int h_out = -1;

  cudaMalloc(&d_out, sizeof(int));
  cudaMemset(d_out, 0xff, sizeof(int));

  dsm_sum_kernel<<<2, 128, sizeof(int)>>>(d_out);
  cudaDeviceSynchronize();

  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  printf("DSM sum = %d\n", h_out);

  cudaFree(d_out);
  return 0;
}
```

编译：

```bash
nvcc -std=c++17 -arch=sm_90a \
  -I/path/to/cutlass/include \
  step2_dsm.cu -o step2_dsm
```

预期输出：

```text
DSM sum = 3
```

### 通过标准

你能自己说清楚：

* DSM 不是“新的独立内存”，而是 cluster 范围内可互访的 shared memory
* `cluster.sync()` 在 DSM 里不是可选项，而是正确性要求

---

# Step 3：只学一个新特性 —— mbarrier

这一阶段只学 Hopper 的**异步同步原语 mbarrier**。先不碰 TMA，只做最小的“生产者写 SMEM，消费者等 barrier，再读 SMEM”。PTX ISA 里专门给了 `mbarrier.init / expect_tx / complete_tx / test_wait / try_wait` 等语义；CUDA Programming Guide 也给出了基于 `__mbarrier_t` 的 producer-consumer 例子。([NVIDIA Docs][4])

### 你要学会什么

1. barrier 不是 `__syncthreads()`
2. 它能表达“某一批异步工作完成了”
3. 以后 TMA 完成通知就是挂在 mbarrier 上的

### 最小可执行代码

这个例子只做一个 block 内的单生产者/多消费者同步。

```cpp
// step3_mbarrier.cu
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_awbarrier_primitives.h>
#include <cute/tensor.hpp>

using namespace cute;

__global__ void mbarrier_demo_kernel(int* out) {
  __shared__ int smem[1];
  __shared__ __mbarrier_t bar;

  if (threadIdx.x == 0) {
    __mbarrier_init(&bar, 1);   // 1 个到达者
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    smem[0] = 42;
    __mbarrier_arrive(&bar);
  } else if (threadIdx.x == 1) {
    // 简单 busy wait
    while (!__mbarrier_test_wait(&bar)) { }
    Tensor gOut = make_tensor(make_gmem_ptr(out), make_shape(Int<1>{}));
    gOut(0) = smem[0];
  }
}

int main() {
  int* d_out;
  int h_out = -1;

  cudaMalloc(&d_out, sizeof(int));
  cudaMemset(d_out, 0xff, sizeof(int));

  mbarrier_demo_kernel<<<1, 32>>>(d_out);
  cudaDeviceSynchronize();

  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  printf("mbarrier observed = %d\n", h_out);

  cudaFree(d_out);
  return 0;
}
```

编译：

```bash
nvcc -std=c++17 -arch=sm_90a \
  -I/path/to/cutlass/include \
  step3_mbarrier.cu -o step3_mbarrier
```

预期输出：

```text
mbarrier observed = 42
```

### 通过标准

你能清楚解释：

* `__syncthreads()` 是“大家到齐”
* `mbarrier` 是“某个事件完成”
* TMA/WGMMA 管线里真正关键的是后者，不是前者

---

# Step 4：只学一个新特性 —— TMA 的“描述符 + 坐标”思维

这一阶段我建议**先不急着发真正的 TMA copy 指令**，先吃透 CuTe 对 TMA 的抽象：**TMA 不是拿一个普通 gmem pointer 去搬，而是 host 先创建 descriptor，kernel 里给 descriptor + smem 指针 + 坐标**。官方 CuTe TMA 文档把这点讲得很清楚：TMA 面向的是多维 tensor descriptor，CuTe 因此要构造“隐式的 TMA 坐标 tensor”，而不是普通指针 tensor。([NVIDIA Docs][5])

### 你要学会什么

1. 为什么普通 `gmem ptr + layout` 不够
2. 为什么 CuTe 会出现 `ArithTuple` / `ArithmeticTupleIterator`
3. TMA 看到的是“坐标”，不是“普通地址偏移”

### 最小可执行代码

这个程序是 host-only，可直接运行。它不是硬件 TMA copy，但它是**理解 CuTe TMA tensor 的最小程序**。

```cpp
// step4_tma_coordinate_tensor.cu
#include <iostream>
#include <cute/tensor.hpp>
#include <cute/numeric/arithmetic_tuple.hpp>

using namespace cute;

int main() {
  // 一个普通 CuTe tensor：逻辑坐标 -> 线性偏移
  auto normal = make_tensor(counting_iterator<int>(0), make_shape(4, 5));
  print(normal);

  std::cout << "----\n";

  // 一个“坐标型 iterator”：逻辑坐标 -> 多维 TMA 坐标
  auto citer = make_inttuple_iter(0, Int<2>{}, Int<7>{});
  auto coord_tensor = make_tensor(
      citer,
      make_layout(make_shape(Int<2>{}, Int<3>{}),
                  make_stride(E<0>{}, E<1>{}))
  );

  print(coord_tensor);

  std::cout << "coord_tensor(1,2) = ";
  print(coord_tensor(1,2));
  std::cout << "\n";

  return 0;
}
```

编译：

```bash
nvcc -std=c++17 -arch=sm_90a \
  -I/path/to/cutlass/include \
  step4_tma_coordinate_tensor.cu -o step4_tma_coordinate_tensor
```

### 通过标准

你能自己回答：

* 为什么 TMA tensor 打印出来会长得很奇怪
* `E<0>{}, E<1>{}` 本质上在干什么
* 为什么 CUTLASS/CuTe 的 TMA 代码里经常先构 descriptor，再去切 tile/partition

---

# Step 5：只学一个新特性 —— WGMMA

这一阶段才上 Hopper 最核心的算子：**WGMMA**。它是 warpgroup 级别的 MMA，按 PTX 和 Colfax 教程的说法，**4 个连续 warp，也就是 128 线程，共同执行一个 `wgmma.mma_async`**；而且 `wgmma.commit_group` / `wait_group` 属于这套异步流水的控制指令，PTX 里标明需要 `sm_90a`。([Colfax Research][6])

### 你要学会什么

1. WGMMA 的执行粒度是 warpgroup，不是 warp
2. WGMMA 是 async 的，要有 commit/wait group 概念
3. 初学时不要手写 PTX，先让 CUTLASS/CuTe 帮你选合法的 `TiledMMA`

### 最小可执行代码

这一步最稳的做法，是用 CUTLASS 3.x/4.x 的 **CuTe-based Hopper GEMM**，让库在 Hopper 上生成 WGMMA 路径。下面这个程序是“最小能跑”的 Hopper GEMM 骨架，重点是看 `CollectiveBuilder`、`TileShape`、`ClusterShape` 这些 CuTe/CUTLASS 层级，而不是一开始去抄 PTX。CUTLASS 官方 quickstart 也正是推荐从 `GemmUniversalAdapter` 这一层入手。([NVIDIA Docs][3])

```cpp
// step5_wgmma_gemm.cu
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementD = float;
using ElementAcc = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using ArchTag = cutlass::arch::Sm90;
using OpClass = cutlass::arch::OpClassTensorOp;

using TileShape = cute::Shape<cute::_128, cute::_128, cute::_64>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

using CollectiveMainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA, 8,
    ElementB, LayoutB, 8,
    ElementAcc,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::KernelScheduleAuto
  >::CollectiveOp;

using CollectiveEpilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc,
    ElementC, LayoutC, 4,
    ElementD, LayoutD, 4,
    cutlass::epilogue::thread::LinearCombination<ElementD, 1, ElementAcc, ElementAcc>
  >::CollectiveOp;

using GemmKernel =
  cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
  >;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

int main() {
  int M = 128, N = 128, K = 64;

  std::vector<ElementA> hA(M*K, ElementA(1));
  std::vector<ElementB> hB(K*N, ElementB(1));
  std::vector<ElementC> hC(M*N, 0.0f);
  std::vector<ElementD> hD(M*N, -1.0f);

  ElementA *dA;
  ElementB *dB;
  ElementC *dC;
  ElementD *dD;

  cudaMalloc(&dA, sizeof(ElementA) * hA.size());
  cudaMalloc(&dB, sizeof(ElementB) * hB.size());
  cudaMalloc(&dC, sizeof(ElementC) * hC.size());
  cudaMalloc(&dD, sizeof(ElementD) * hD.size());

  cudaMemcpy(dA, hA.data(), sizeof(ElementA) * hA.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB.data(), sizeof(ElementB) * hB.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dC, hC.data(), sizeof(ElementC) * hC.size(), cudaMemcpyHostToDevice);

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    {dA, K},
    {dB, K},
    {dC, N},
    {dD, N},
    {1.0f, 0.0f}
  };

  Gemm gemm_op;
  size_t workspace_size = Gemm::get_workspace_size(args);
  void* workspace = nullptr;
  cudaMalloc(&workspace, workspace_size);

  cutlass::Status st = gemm_op.initialize(args, workspace);
  if (st != cutlass::Status::kSuccess) {
    std::cerr << "initialize failed\n";
    return -1;
  }

  st = gemm_op();
  if (st != cutlass::Status::kSuccess) {
    std::cerr << "run failed\n";
    return -1;
  }

  cudaMemcpy(hD.data(), dD, sizeof(ElementD) * hD.size(), cudaMemcpyDeviceToHost);
  std::cout << "D[0] = " << hD[0] << "\n";   // 预期约为 64

  cudaFree(workspace);
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
  return 0;
}
```

编译：

```bash
nvcc -std=c++17 -arch=sm_90a \
  -I/path/to/cutlass/include \
  -I/path/to/cutlass/tools/util/include \
  step5_wgmma_gemm.cu -o step5_wgmma_gemm
```

### 通过标准

你能解释：

* `TileShape<M,N,K>` 是 CTA 级 tile
* `ClusterShape` 是 cluster 级 tile
* `CollectiveMainloop` 里真正决定 Hopper Tensor Core 路径
* WGMMA 的真实细节暂时先交给 CUTLASS/CuTe，不手写 PTX

---

# Step 6：只学一个新特性 —— Warp Specialization

这一阶段加上 Hopper GEMM 的真正灵魂：**producer warp group 和 consumer warp group 分工**。CUTLASS 官方“Efficient GEMM in CUDA”明确写到，Hopper 从 CUTLASS 3.0 开始，把线程块拆成 producer warp group 用 TMA 从 gmem 搬到 smem，consumer warp group 用 WGMMA 做 Tensor Core 计算。([NVIDIA Docs][2])

### 你要学会什么

1. 为什么 producer/consumer 分工后 Tensor Core 更不容易饿死
2. 为什么这一步必须把 TMA、mbarrier、WGMMA 串起来
3. 为什么初学时最好基于官方 example 50 裁剪，而不是自己从零拼 `CollectiveMma`

### 最小可执行代码

这一步最推荐的做法是：**直接从 CUTLASS 官方 `examples/50_hopper_gemm_with_epilogue_swizzle` 裁剪**。官方这个例子本身就是 Hopper 自定义 collective mainloop，并且源码里明确写了：

* `MainloopSm90TmaGmmaWarpSpecialized`
* `KernelTmaWarpSpecialized`
* `cute::GMMA::ss_op_selector`
* `cute::SM90_TMA_LOAD` / `SM90_TMA_LOAD_MULTICAST`

这些正好对应你前面学过的所有点。源码也说明它是 Hopper GEMM 的定制 collective 示例。([GitHub][7])

你不必整份抄下来。只需要在 Step 5 的程序里，把 `CollectiveMainloop` 换成下面这个“显式 warp-specialized 版本”；其余主函数、初始化、launch 逻辑保持不变即可。

```cpp
// 只替换 Step 5 里的 CollectiveMainloop 相关定义

using TileShape = cute::Shape<cute::_128, cute::_64, cute::_128>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
constexpr int PipelineStages = 4;

static constexpr cute::GMMA::Major GmmaMajorA = cute::GMMA::Major::K;
static constexpr cute::GMMA::Major GmmaMajorB = cute::GMMA::Major::K;

using DispatchPolicy =
  cutlass::gemm::MainloopSm90TmaGmmaWarpSpecialized<
    PipelineStages,
    ClusterShape,
    cutlass::gemm::KernelTmaWarpSpecialized
  >;

using TiledMma =
  decltype(cute::make_tiled_mma(
    cute::GMMA::ss_op_selector<
      ElementA, ElementB, ElementAcc,
      TileShape, GmmaMajorA, GmmaMajorB
    >()
  ));

using SmemLayoutAtomA =
  decltype(cutlass::gemm::collective::detail::ss_smem_selector<
    GmmaMajorA, ElementA,
    decltype(cute::get<0>(TileShape{})),
    decltype(cute::get<2>(TileShape{}))>());

using SmemLayoutAtomB =
  decltype(cutlass::gemm::collective::detail::ss_smem_selector<
    GmmaMajorB, ElementB,
    decltype(cute::get<1>(TileShape{})),
    decltype(cute::get<2>(TileShape{}))>());

using CollectiveMainloop =
  cutlass::gemm::collective::CollectiveMma<
    DispatchPolicy,
    TileShape,
    ElementA, cutlass::gemm::TagToStrideA_t<LayoutA>,
    ElementB, cutlass::gemm::TagToStrideB_t<LayoutB>,
    TiledMma,
    cute::SM90_TMA_LOAD,
    SmemLayoutAtomA,
    void,
    cute::identity,
    cute::SM90_TMA_LOAD,
    SmemLayoutAtomB,
    void,
    cute::identity
  >;
```

### 通过标准

你能清楚说出：

* Step 5 是“先跑起来的 Hopper Tensor Core GEMM”
* Step 6 才是“真正 Hopper 风格”的 kernel 组织方式
* `TMA + mbarrier + WGMMA + producer/consumer` 才是 Hopper kernel 的主旋律

---

# 我建议的学习节奏

按最现实的节奏：

1. **第 1 天**：Step 1 + Step 2
   只把 cluster / DSM 跑通。
2. **第 2 天**：Step 3
   把 mbarrier 的“事件完成”语义吃透。
3. **第 3 天**：Step 4
   把 CuTe 的 TMA tensor 和 `ArithTuple` 看明白。
4. **第 4~5 天**：Step 5
   跑通一个 Hopper CUTLASS GEMM。
5. **第 6~7 天**：Step 6
   对照官方 example 50，把 warp specialization 的 mainloop 摸透。

这条顺序基本就是从 CUDA guide 的 Hopper 特性，到 CuTe tensor/model，再到 CUTLASS Hopper GEMM 的自然递进；官方文档、CUTLASS 例子和 Hopper GEMM 教程本质上也都是沿着这条主线组织的。([NVIDIA Docs][1])

---

# 最后给你一个很重要的提醒

**不要一开始就钻 TMA/WGMMA 的 PTX 细节。**
对初学者最有效的策略是：

* 先用 CuTe 看懂 layout / tensor / partition
* 再把 Hopper 的 cluster、DSM、mbarrier 概念跑通
* 再用 CUTLASS 的 collective 层去接触 WGMMA
* 最后再回头看 `wgmma.mma_async`、`commit_group`、`wait_group` 和 TMA descriptor

因为官方资料本身已经说明，Hopper 高性能 GEMM 是一套**层次化设计**，不是单条指令技巧。你先学层次，后学指令，效率最高。([GitHub][8])

下一步最合适的是把 **Step 5 的程序改成一个“纯 CuTe 风格的最小 WGMMA tile kernel”**，我建议就从 `TiledMMA`、`partition_fragment_C`、`gemm(tiled_mma, ...)` 这一层开始。

[1]: https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf "https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf"
[2]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html"
[3]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html "Quickstart — NVIDIA CUTLASS Documentation"
[4]: https://docs.nvidia.com/cuda/parallel-thread-execution/ "1. Introduction — PTX ISA 9.2 documentation"
[5]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0z_tma_tensors.html "CuTe TMA Tensors — NVIDIA CUTLASS Documentation"
[6]: https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/ "CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on NVIDIA® Hopper™ GPUs - Colfax Research"
[7]: https://github.com/NVIDIA/cutlass/blob/main/examples/50_hopper_gemm_with_epilogue_swizzle/50_hopper_gemm_with_epilogue_swizzle.cu "cutlass/examples/50_hopper_gemm_with_epilogue_swizzle/50_hopper_gemm_with_epilogue_swizzle.cu at main · NVIDIA/cutlass · GitHub"
[8]: https://github.com/NVIDIA/cutlass "https://github.com/NVIDIA/cutlass"
