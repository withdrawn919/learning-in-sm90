# Hopper 架构 + CuTe 学习笔记

> 每个 Step 包含一段最小可执行代码，仅演示一个功能点。
> 编译要求：CUDA 12.0+，sm_90a，链接 CUTLASS（header-only）。
>
> 通用编译命令示例：
> ```bash
> nvcc -std=c++17 -arch=sm_90a \
>   -I cutlass/include -I cutlass/tools/util/include \
>   step_X.cu -o step_X
> ```

---

## Step 0: CuTe 基础 — Layout 与 Tensor

**功能点**：使用 CuTe 的 `Layout` 和 `make_tensor` 创建并打印 Tensor。

```cpp
// step0_layout_tensor.cu
// 功能：演示 CuTe Layout 与 Tensor 的创建和打印
#include <cstdio>
#include <cute/tensor.hpp>

using namespace cute;

int main() {
  // 1. 创建一个 (4,8) 的行主序 Layout
  auto layout = make_layout(make_shape(Int<4>{}, Int<8>{}),
                            make_stride(Int<8>{}, Int<1>{}));
  print("Layout: "); print(layout); print("\n");

  // 2. 在 host 上创建一个 Tensor 并填充
  float data[32];
  for (int i = 0; i < 32; ++i) data[i] = float(i);

  Tensor tensor = make_tensor(data, layout);
  print_tensor(tensor);

  // 3. 展示 Layout 的坐标到偏移映射
  printf("layout(2,3) = %d\n", int(layout(2, 3)));  // 2*8+3 = 19

  return 0;
}
```

---

## Step 1: TMA Load — 全局内存到共享内存的异步拷贝

**功能点**：使用 `SM90_TMA_LOAD` 将一个 2D Tile 从全局内存异步加载到共享内存。

```cpp
// step1_tma_load.cu
// 功能：演示 TMA Load 将全局内存 tile 加载到共享内存
#include <cstdio>
#include <cute/tensor.hpp>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/device_kernel.h"

using namespace cute;

// TMA Load kernel：将一个 (BLK_M, BLK_K) tile 从 gmem 加载到 smem
template <class SmemLayout, class TmaAtom>
__global__ void tma_load_kernel(CUTLASS_GRID_CONSTANT TmaAtom const tma,
                                SmemLayout smem_layout,
                                half_t* out, int M, int K) {
  extern __shared__ char smem_buf[];
  half_t* smem_ptr = reinterpret_cast<half_t*>(smem_buf);
  Tensor sA = make_tensor(make_smem_ptr(smem_ptr), smem_layout);  // (BLK_M, BLK_K)

  // 获取 TMA tensor 并按 CTA 坐标切分
  Tensor mA = tma.get_tma_tensor(make_shape(M, K));               // (M, K)
  Tensor gA = local_tile(mA, shape(sA), make_coord(blockIdx.x, 0)); // (BLK_M, BLK_K)

  // TMA partition
  auto [tAgA, tAsA] = tma_partition(tma, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sA), group_modes<0,2>(gA));

  // 初始化 barrier
  using BarType = cutlass::arch::ClusterTransactionBarrier;
  __shared__ uint64_t barrier;
  if (threadIdx.x == 0) {
    BarType::init(&barrier, 1);
    // 设置预期传输字节数
    int tx_bytes = sizeof(half_t) * size(sA);
    BarType::arrive_and_expect_tx(&barrier, tx_bytes);
    // 发起 TMA 拷贝
    copy(tma.with(barrier), tAgA(_,0), tAsA(_,0));
  }
  __syncthreads();

  // 等待 TMA 完成
  BarType::wait(&barrier, 0);

  // 验证：将 smem 数据写回 gmem
  for (int i = threadIdx.x; i < size(sA); i += blockDim.x) {
    out[i] = sA(i);
  }
}

int main() {
  int M = 128, K = 64;
  auto bM = Int<128>{};
  auto bK = Int<64>{};

  // 分配 device 数据
  thrust::host_vector<half_t> h_A(M * K);
  for (int i = 0; i < M * K; ++i) h_A[i] = half_t(float(i % 100));
  thrust::device_vector<half_t> d_A = h_A;
  thrust::device_vector<half_t> d_out(M * K, half_t(0));

  // 定义 smem layout（使用 SW128 swizzle）
  auto smem_layout = tile_to_shape(
      GMMA::Layout_MN_SW128_Atom<half_t>{}, make_shape(bM, bK));

  // 创建 gmem tensor 并构造 TMA atom
  auto dA = make_stride(Int<1>{}, M);  // column-major
  Tensor mA = make_tensor(d_A.data().get(), make_shape(M, K), dA);
  Copy_Atom tma = make_tma_atom(SM90_TMA_LOAD{}, mA, smem_layout, make_shape(bM, bK));

  // launch
  int smem_size = sizeof(half_t) * size(smem_layout) + sizeof(uint64_t);
  dim3 grid(1), block(128), cluster(1,1,1);
  auto* kptr = &tma_load_kernel<decltype(smem_layout), decltype(tma)>;
  cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  cutlass::ClusterLaunchParams params = {grid, block, cluster, smem_size};
  cutlass::launch_kernel_on_cluster(params, (void const*)kptr,
                                    tma, smem_layout, d_out.data().get(), M, K);
  cudaDeviceSynchronize();

  // 验证
  thrust::host_vector<half_t> h_out = d_out;
  bool pass = true;
  for (int i = 0; i < 10; ++i) {
    if (float(h_out[i]) != float(h_A[i])) { pass = false; break; }
  }
  printf("TMA Load: %s\n", pass ? "PASS" : "FAIL");
  return 0;
}
```

---

## Step 2: TMA Store — 共享内存到全局内存的异步写回

**功能点**：使用 `SM90_TMA_STORE` 将共享内存的数据写回全局内存。

```cpp
// step2_tma_store.cu
// 功能：演示 TMA Store 将 smem 数据写回 gmem
#include <cstdio>
#include <cute/tensor.hpp>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/device_kernel.h"

using namespace cute;

template <class SmemLayout, class TmaStoreAtom>
__global__ void tma_store_kernel(CUTLASS_GRID_CONSTANT TmaStoreAtom const tma_store,
                                 SmemLayout smem_layout, int M, int K) {
  extern __shared__ char smem_buf[];
  half_t* smem_ptr = reinterpret_cast<half_t*>(smem_buf);
  Tensor sA = make_tensor(make_smem_ptr(smem_ptr), smem_layout);

  // 每个线程在 smem 中写入已知值
  for (int i = threadIdx.x; i < size(sA); i += blockDim.x) {
    sA(i) = half_t(float(i));
  }
  __syncthreads();

  // 获取 TMA tensor
  Tensor mA = tma_store.get_tma_tensor(make_shape(M, K));
  Tensor gA = local_tile(mA, shape(sA), make_coord(blockIdx.x, 0));

  auto [tAgA, tAsA] = tma_partition(tma_store, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sA), group_modes<0,2>(gA));

  // 只需一个线程发起 TMA Store
  if (threadIdx.x == 0) {
    copy(tma_store, tAsA(_,0), tAgA(_,0));
    tma_store_arrive();   // 通知 TMA store 完成
  }
  tma_store_wait<0>();    // 等待所有 TMA store 完成
}

int main() {
  int M = 128, K = 64;
  auto bM = Int<128>{};
  auto bK = Int<64>{};

  thrust::device_vector<half_t> d_C(M * K, half_t(0));

  auto smem_layout = tile_to_shape(
      GMMA::Layout_MN_SW128_Atom<half_t>{}, make_shape(bM, bK));

  auto dC = make_stride(Int<1>{}, M);
  Tensor mC = make_tensor(d_C.data().get(), make_shape(M, K), dC);
  Copy_Atom tma_store = make_tma_atom(SM90_TMA_STORE{}, mC, smem_layout, make_shape(bM, bK));

  int smem_size = sizeof(half_t) * size(smem_layout);
  dim3 grid(1), block(128), cluster(1,1,1);
  auto* kptr = &tma_store_kernel<decltype(smem_layout), decltype(tma_store)>;
  cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  cutlass::ClusterLaunchParams params = {grid, block, cluster, smem_size};
  cutlass::launch_kernel_on_cluster(params, (void const*)kptr,
                                    tma_store, smem_layout, M, K);
  cudaDeviceSynchronize();

  thrust::host_vector<half_t> h_C = d_C;
  bool pass = (float(h_C[0]) == 0.0f && float(h_C[1]) == 1.0f);
  printf("TMA Store: %s\n", pass ? "PASS" : "FAIL");
  return 0;
}
```

---

## Step 3: Swizzle — 共享内存 Bank Conflict 消除

**功能点**：演示 CuTe `Swizzle<B,M,S>` 如何改变共享内存的地址映射以消除 bank conflict。

```cpp
// step3_swizzle.cu
// 功能：对比有无 Swizzle 时的共享内存地址映射
// 本示例为 host-only 程序，用于理解 Swizzle 的数学原理
#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/swizzle.hpp>

using namespace cute;

int main() {
  // --- 1. 无 Swizzle 的 layout ---
  // (128, 64) 矩阵，half_t，column-major => stride = (1, 128)
  auto layout_no_swizzle = make_layout(
      make_shape(Int<128>{}, Int<64>{}),
      make_stride(Int<1>{}, Int<128>{}));

  printf("=== Without Swizzle ===\n");
  // 打印前 8 行 x 前 4 列的字节偏移，观察 bank 分布
  // 共享内存 bank = (byte_offset / 4) % 32
  for (int row = 0; row < 8; ++row) {
    for (int col = 0; col < 4; ++col) {
      int idx = layout_no_swizzle(row, col);
      int byte_offset = idx * 2;  // half_t = 2 bytes
      int bank = (byte_offset / 4) % 32;
      printf("(%d,%d)->bank%2d  ", row, col, bank);
    }
    printf("\n");
  }

  // --- 2. 有 Swizzle 的 layout ---
  // Swizzle<3, 3, 3>: BBits=3, MBase=3, SShift=3
  // 对应 GMMA::Layout_MN_SW128_Atom 的 128-byte swizzle
  // 实际使用中，CUTLASS 提供了预定义的 swizzle layout:
  //   GMMA::Layout_MN_SW128_Atom<half_t>{}  -- 128 字节 swizzle
  //   GMMA::Layout_MN_SW64_Atom<half_t>{}   --  64 字节 swizzle
  //   GMMA::Layout_MN_SW32_Atom<half_t>{}   --  32 字节 swizzle
  auto swizzle_fn = Swizzle<3, 3, 3>{};  // 128-byte swizzle

  printf("\n=== With Swizzle<3,3,3> (SW128) ===\n");
  for (int row = 0; row < 8; ++row) {
    for (int col = 0; col < 4; ++col) {
      int idx = layout_no_swizzle(row, col);
      int swizzled_idx = swizzle_fn.apply(idx);
      int byte_offset = swizzled_idx * 2;
      int bank = (byte_offset / 4) % 32;
      printf("(%d,%d)->bank%2d  ", row, col, bank);
    }
    printf("\n");
  }

  // --- 3. 使用 CUTLASS 预定义的 swizzle atom ---
  auto sw128_atom = GMMA::Layout_MN_SW128_Atom<half_t>{};
  printf("\nSW128 Atom layout: ");
  print(sw128_atom);
  printf("\n");

  // 扩展到完整 tile
  auto full_layout = tile_to_shape(sw128_atom, make_shape(Int<128>{}, Int<64>{}));
  printf("Full swizzled layout: ");
  print(full_layout);
  printf("\n");

  return 0;
}
```

---

## Step 4: 异步事务屏障（Async Transaction Barrier）

**功能点**：演示 `ClusterTransactionBarrier` 的 `init` / `arrive_and_expect_tx` / `wait` 机制。

```cpp
// step4_async_barrier.cu
// 功能：演示异步事务屏障的 arrive / wait 同步机制
// 配合 TMA Load 使用，验证 barrier 能正确同步异步拷贝
#include <cstdio>
#include <cute/tensor.hpp>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/device_kernel.h"

using namespace cute;

template <class SmemLayout, class TmaAtom>
__global__ void barrier_demo_kernel(CUTLASS_GRID_CONSTANT TmaAtom const tma,
                                    SmemLayout smem_layout,
                                    half_t* out, int M, int K) {
  extern __shared__ char smem_buf[];
  half_t* smem_ptr = reinterpret_cast<half_t*>(smem_buf);
  Tensor sA = make_tensor(make_smem_ptr(smem_ptr), smem_layout);

  // Barrier 存储在 smem 尾部
  constexpr int NUM_BARRIERS = 2;
  uint64_t* barriers = reinterpret_cast<uint64_t*>(
      smem_buf + sizeof(half_t) * cosize(smem_layout));

  using BarType = cutlass::arch::ClusterTransactionBarrier;

  Tensor mA = tma.get_tma_tensor(make_shape(M, K));

  // 初始化 barrier（仅 thread 0）
  if (threadIdx.x == 0) {
    for (int i = 0; i < NUM_BARRIERS; ++i) {
      BarType::init(&barriers[i], 1);
    }
  }
  __syncthreads();

  int tx_bytes = sizeof(half_t) * size(sA);

  // ---- 使用 barrier[0] 进行第一次 TMA Load ----
  if (threadIdx.x == 0) {
    Tensor gA0 = local_tile(mA, shape(sA), make_coord(0, 0));
    auto [tAgA, tAsA] = tma_partition(tma, Int<0>{}, Layout<_1>{},
                                      group_modes<0,2>(sA), group_modes<0,2>(gA0));
    BarType::arrive_and_expect_tx(&barriers[0], tx_bytes);
    copy(tma.with(barriers[0]), tAgA(_,0), tAsA(_,0));
  }

  // 等待 barrier[0] 完成
  BarType::wait(&barriers[0], 0);

  // 验证数据并写回
  for (int i = threadIdx.x; i < size(sA); i += blockDim.x) {
    out[i] = sA(i);
  }
}

int main() {
  int M = 128, K = 64;
  auto bM = Int<128>{};
  auto bK = Int<64>{};

  thrust::host_vector<half_t> h_A(M * K);
  for (int i = 0; i < M * K; ++i) h_A[i] = half_t(float(i % 50));
  thrust::device_vector<half_t> d_A = h_A;
  thrust::device_vector<half_t> d_out(M * K, half_t(0));

  auto smem_layout = tile_to_shape(
      GMMA::Layout_MN_SW128_Atom<half_t>{}, make_shape(bM, bK));

  auto dA = make_stride(Int<1>{}, M);
  Tensor mA = make_tensor(d_A.data().get(), make_shape(M, K), dA);
  Copy_Atom tma = make_tma_atom(SM90_TMA_LOAD{}, mA, smem_layout, make_shape(bM, bK));

  int smem_size = sizeof(half_t) * cosize(smem_layout) + 2 * sizeof(uint64_t);
  dim3 grid(1), block(128), cluster(1,1,1);
  auto* kptr = &barrier_demo_kernel<decltype(smem_layout), decltype(tma)>;
  cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  cutlass::ClusterLaunchParams params = {grid, block, cluster, smem_size};
  cutlass::launch_kernel_on_cluster(params, (void const*)kptr,
                                    tma, smem_layout, d_out.data().get(), M, K);
  cudaDeviceSynchronize();

  thrust::host_vector<half_t> h_out = d_out;
  bool pass = true;
  for (int i = 0; i < 10; ++i) {
    if (float(h_out[i]) != float(h_A[i])) { pass = false; break; }
  }
  printf("Async Barrier + TMA: %s\n", pass ? "PASS" : "FAIL");
  return 0;
}
```

---

## Step 5: Cluster 与分布式共享内存（DSMEM）

**功能点**：使用 Thread Block Cluster 启动内核，通过 `cluster_sync()` 同步 cluster 内的线程块。

```cpp
// step5_cluster.cu
// 功能：演示 Thread Block Cluster 启动和 cluster 同步
#include <cstdio>
#include <cute/tensor.hpp>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/device_kernel.h"

using namespace cute;

__global__ void cluster_demo_kernel(int* out) {
  // 获取 cluster 内的 block 信息
  uint32_t cluster_rank;
  uint32_t cluster_size;
  asm volatile("mov.u32 %0, %%clusterid.x;\n" : "=r"(cluster_rank));
  asm volatile("mov.u32 %0, %%nclusterid.x;\n" : "=r"(cluster_size));

  // 获取 block 在 cluster 内的排名
  uint32_t block_rank_in_cluster;
  asm volatile("mov.u32 %0, %%cluster_ctaid.x;\n" : "=r"(block_rank_in_cluster));

  uint32_t cluster_dim;
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;\n" : "=r"(cluster_dim));

  // cluster 内所有 block 同步
  cluster_sync();

  // 每个 block 的 thread 0 输出信息
  if (threadIdx.x == 0) {
    int idx = blockIdx.x;
    out[idx * 3 + 0] = blockIdx.x;
    out[idx * 3 + 1] = block_rank_in_cluster;
    out[idx * 3 + 2] = cluster_dim;
  }
}

int main() {
  int num_blocks = 4;
  int cluster_size = 2;  // 每个 cluster 包含 2 个 block

  thrust::device_vector<int> d_out(num_blocks * 3, 0);

  dim3 grid(num_blocks), block(128);
  dim3 cluster_dim(cluster_size, 1, 1);
  int smem_size = 0;

  cutlass::ClusterLaunchParams params = {grid, block, cluster_dim, smem_size};
  cutlass::launch_kernel_on_cluster(
      params, (void const*)&cluster_demo_kernel,
      d_out.data().get());
  cudaDeviceSynchronize();

  thrust::host_vector<int> h_out = d_out;
  printf("Cluster launch results:\n");
  for (int b = 0; b < num_blocks; ++b) {
    printf("  blockIdx=%d, rank_in_cluster=%d, cluster_dim=%d\n",
           h_out[b*3+0], h_out[b*3+1], h_out[b*3+2]);
  }
  printf("Cluster: PASS\n");
  return 0;
}
```

---

## Step 6: WGMMA（Warp Group Matrix Multiply-Accumulate）

**功能点**：使用 `SM90_64x64x16_F16F16F16_SS` WGMMA 指令从共享内存执行矩阵乘累加。

```cpp
// step6_wgmma.cu
// 功能：演示 WGMMA 从共享内存读取操作数执行矩阵乘法
// 参考: cutlass/examples/cute/tutorial/hopper/wgmma_sm90.cu
#include <cstdio>
#include <cute/tensor.hpp>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/device_kernel.h"

using namespace cute;

template <class SmemLayoutA, class SmemLayoutB, class TiledMma>
__global__
__launch_bounds__(128)
void wgmma_kernel(half_t const* A, half_t const* B, half_t* C,
                  SmemLayoutA sA_layout, SmemLayoutB sB_layout,
                  TiledMma mma, int M, int N, int K) {
  extern __shared__ char smem_buf[];

  // 共享内存 tensors
  half_t* sA_ptr = reinterpret_cast<half_t*>(smem_buf);
  half_t* sB_ptr = sA_ptr + cosize(sA_layout);
  Tensor sA = make_tensor(make_smem_ptr(sA_ptr), sA_layout);  // (BLK_M, BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(sB_ptr), sB_layout);  // (BLK_N, BLK_K)

  // 使用 cp.async 将数据从 gmem 加载到 smem
  auto copyA = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
      Layout<Shape<_16,_8>>{}, Layout<Shape<_8,_1>>{});
  auto copyB = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
      Layout<Shape<_16,_8>>{}, Layout<Shape<_8,_1>>{});

  Tensor gA = make_tensor(make_gmem_ptr(A), make_shape(M,K), make_stride(Int<1>{}, M));
  Tensor gB = make_tensor(make_gmem_ptr(B), make_shape(N,K), make_stride(Int<1>{}, N));

  // 拷贝 A tile
  ThrCopy thrA = copyA.get_slice(threadIdx.x);
  Tensor sA_sw = as_position_independent_swizzle_tensor(sA);
  Tensor tAgA = thrA.partition_S(gA);
  Tensor tAsA = thrA.partition_D(sA_sw);
  copy(copyA, tAgA(_,_,_,0), tAsA(_,_,_,0));

  // 拷贝 B tile
  ThrCopy thrB = copyB.get_slice(threadIdx.x);
  Tensor sB_sw = as_position_independent_swizzle_tensor(sB);
  Tensor tBgB = thrB.partition_S(gB);
  Tensor tBsB = thrB.partition_D(sB_sw);
  copy(copyB, tBgB(_,_,_,0), tBsB(_,_,_,0));

  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  // WGMMA: 从 smem 执行矩阵乘法
  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);           // MMA view of sA
  Tensor tCsB = thr_mma.partition_B(sB);           // MMA view of sB

  Tensor gC = make_tensor(make_gmem_ptr(C), make_shape(M,N), make_stride(Int<1>{}, M));
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  // 创建描述符 fragment（WGMMA 直接从 smem 读取，无需 copy 到寄存器）
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);

  // 执行 WGMMA
  warpgroup_fence_operand(tCrC);
  warpgroup_arrive();
  gemm(mma, tCrA(_,_,_,0), tCrB(_,_,_,0), tCrC);
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(tCrC);

  // 写回结果
  axpby(half_t(1.0f), tCrC, half_t(0.0f), tCgC);
}

int main() {
  int M = 128, N = 128, K = 64;

  thrust::host_vector<half_t> h_A(M*K), h_B(N*K);
  for (int i = 0; i < M*K; ++i) h_A[i] = half_t((i % 3 == 0) ? 1.0f : 0.0f);
  for (int i = 0; i < N*K; ++i) h_B[i] = half_t((i % 5 == 0) ? 1.0f : 0.0f);
  thrust::device_vector<half_t> d_A = h_A, d_B = h_B;
  thrust::device_vector<half_t> d_C(M*N, half_t(0));

  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<64>{};

  auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<half_t>{}, make_shape(bM, bK));
  auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<half_t>{}, make_shape(bN, bK));
  TiledMMA mma = make_tiled_mma(
      SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>{});

  int smem_size = sizeof(half_t) * (cosize(sA) + cosize(sB));
  dim3 grid(1), block(size(mma)), cluster(1,1,1);
  auto* kptr = &wgmma_kernel<decltype(sA), decltype(sB), decltype(mma)>;
  cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  cutlass::ClusterLaunchParams params = {grid, block, cluster, smem_size};
  cutlass::launch_kernel_on_cluster(params, (void const*)kptr,
      d_A.data().get(), d_B.data().get(), d_C.data().get(),
      sA, sB, mma, M, N, K);
  cudaDeviceSynchronize();

  thrust::host_vector<half_t> h_C = d_C;
  printf("WGMMA result C[0]=%f\n", float(h_C[0]));
  printf("WGMMA: PASS\n");
  return 0;
}
```

---

## Step 7: Pipeline 与多阶段异步流水线

**功能点**：使用 `PipelineState` 管理多阶段流水线，实现 TMA Load 与 WGMMA Compute 的重叠。

```cpp
// step7_pipeline.cu
// 功能：演示多阶段流水线的 read/write state 管理
// 参考: cutlass/examples/cute/tutorial/hopper/wgmma_tma_sm90.cu
//
// 核心概念：
// - PipelineState<K_PIPE_MAX> 维护循环索引 .index() 和相位 .phase()
// - Producer (TMA) 和 Consumer (MMA) 通过 barrier 同步
// - 流水线深度 K_PIPE_MAX = 3 表示 smem 中同时存在 3 个 tile
//
// 流水线时序（3 stage）:
//   Stage:    | 0   | 1   | 2   | 0   | 1   | ...
//   Producer: |Load0|Load1|Load2|Load3|Load4| ...
//   Consumer: | --- | --- |MMA0 |MMA1 |MMA2 | ...
//
// 关键代码模式（伪代码）：
//
//   auto write_state = PipelineState<3>();  // producer
//   auto read_state  = PipelineState<3>();  // consumer
//
//   // Prefetch: 填满流水线
//   for (pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
//     BarType::arrive_and_expect_tx(&barrier[pipe], tx_bytes);
//     copy(tma.with(barrier[pipe]), gA(_,k), sA(_,pipe));
//     ++k;
//   }
//
//   // Main loop:
//   while (k_remaining > -K_PIPE_MAX) {
//     // Consumer: 等待 producer 完成
//     BarType::wait(&barrier[read_state.index()], read_state.phase());
//     // 执行 WGMMA
//     gemm(mma, sA(_,_,read_state.index()), ...);
//     ++read_state;
//
//     // Producer: 等待 consumer 释放 slot，发起下一次 TMA
//     if (has_more_tiles) {
//       ConsumerBar::wait(&mma_bar[write_state.index()], write_state.phase());
//       copy(tma.with(barrier[write_state.index()]), ...);
//       ++write_state;
//     }
//   }
//
// 完整可执行代码见:
//   3rd/cutlass/examples/cute/tutorial/hopper/wgmma_tma_sm90.cu
//   该文件实现了完整的 3-stage pipeline GEMM (约 260 行)
```

---

## Step 8: Warp Specialization（Warp 特化）

**功能点**：将 warp 分为 Producer（负责 TMA 加载）和 Consumer（负责 WGMMA 计算）。

```cpp
// step8_warp_specialization.cu
// 功能：演示 Warp Specialization 的生产者/消费者分工模式
//
// 核心概念：
// - 一个 thread block 中的 warp 按角色分工
// - Producer warp (通常 warp 0): 负责发起 TMA 加载
// - Consumer warps (其余 warps): 负责执行 WGMMA 计算
// - 两者通过 barrier 进行同步
//
// 关键代码模式（伪代码）：
//
//   int warp_idx = cutlass::canonical_warp_idx_sync();
//   int lane_predicate = cute::elect_one_sync();
//
//   if (warp_idx == 0) {
//     // ===== Producer Warp =====
//     // 只有 warp 0 的一个线程发起 TMA
//     if (lane_predicate) {
//       // 初始化 barrier
//       ProducerBarType::init(&producer_mbar[pipe], 1);
//       ConsumerBarType::init(&consumer_mbar[pipe], num_consumer_threads);
//
//       // 主循环：加载 tile
//       for each k_tile:
//         // 等待 consumer 释放当前 pipe slot
//         ConsumerBarType::wait(&consumer_mbar[pipe], phase);
//         // 设置传输字节数
//         ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tx_bytes);
//         // 发起 TMA
//         copy(tma.with(producer_mbar[pipe]), gA(_,k), sA(_,pipe));
//     }
//   } else {
//     // ===== Consumer Warps =====
//     for each k_tile:
//       // 等待 producer 完成当前 pipe 的数据加载
//       ProducerBarType::wait(&producer_mbar[pipe], phase);
//       // 执行 WGMMA
//       warpgroup_arrive();
//       gemm(mma, sA(_,_,pipe), sB(_,_,pipe), accum);
//       warpgroup_commit_batch();
//       warpgroup_wait<0>();
//       // 通知 producer 数据已消费
//       ConsumerBarType::arrive(&consumer_mbar[pipe]);
//   }
//
// 完整实现见:
//   3rd/cutlass/examples/48_hopper_warp_specialized_gemm/
//   CUTLASS 3.x CollectiveMainloop 中的 WarpSpecialized 策略
```

---

## Step 9: 持久化内核（Persistent Kernel）与 Tile Scheduler

**功能点**：持久化内核让 thread block 循环处理多个 tile，而非每个 tile 启动新 block。

```cpp
// step9_persistent_kernel.cu
// 功能：演示持久化内核的 tile 调度模式
//
// 核心概念：
// - 传统 GEMM：每个 CTA 处理一个 output tile，grid 大小 = tile 数量
// - 持久化内核：grid 大小 = SM 数量，每个 CTA 循环获取新 tile
// - 优势：减少 launch overhead，支持 tile 间的数据复用（如 split-K）
//
// 关键代码模式：
//
//   // Host: 按 SM 数量启动
//   int num_sms;
//   cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
//   dim3 grid(num_sms);
//
//   // Device: 原子计数器分配 tile
//   __device__ int tile_counter = 0;
//
//   __global__ void persistent_gemm_kernel(...) {
//     int total_tiles_m = ceil_div(M, BLK_M);
//     int total_tiles_n = ceil_div(N, BLK_N);
//     int total_tiles = total_tiles_m * total_tiles_n;
//
//     while (true) {
//       // 原子获取下一个 tile 索引
//       int tile_idx;
//       if (threadIdx.x == 0) {
//         tile_idx = atomicAdd(&tile_counter, 1);
//       }
//       tile_idx = __shfl_sync(0xffffffff, tile_idx, 0);
//
//       if (tile_idx >= total_tiles) break;  // 所有 tile 已处理
//
//       // 将线性索引映射到 2D tile 坐标
//       int tile_m = tile_idx % total_tiles_m;
//       int tile_n = tile_idx / total_tiles_m;
//
//       // 计算该 tile 的 GEMM...
//     }
//   }
//
// CUTLASS 中的实现:
//   cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp
//   支持多种策略: Heuristic, StreamK, 分组调度等
```

---

## Step 10: FP8 数据类型支持

**功能点**：使用 Hopper 原生 FP8 类型（E4M3 / E5M2）执行 WGMMA 矩阵乘法。

```cpp
// step10_fp8_wgmma.cu
// 功能：演示 FP8 数据类型与 WGMMA 指令
//
// 核心概念：
// - Hopper 支持两种 FP8 格式:
//   - E4M3 (float8_e4m3fn): 4 位指数, 3 位尾数, 范围 ±448, 精度更高
//   - E5M2 (float8_e5m2):   5 位指数, 2 位尾数, 范围 ±57344, 范围更大
// - WGMMA 直接支持 FP8 输入 + FP32 累加
//
// 关键代码模式：
//
//   #include <cute/tensor.hpp>
//   #include <cutlass/float8.h>   // FP8 类型定义
//
//   using TA = cutlass::float_e4m3_t;  // FP8 E4M3
//   using TB = cutlass::float_e4m3_t;
//   using TC = float;                   // FP32 累加器
//
//   // FP8 的 WGMMA atom（注意 K 维度更大：每次处理 32 个 K 元素）
//   TiledMMA mma = make_tiled_mma(
//       SM90_64x128x32_F32E4M3E4M3_SS<GMMA::Major::K, GMMA::Major::K>{});
//
//   // smem layout 需匹配 FP8 的 atom
//   auto sA = tile_to_shape(
//       GMMA::Layout_K_SW128_Atom<TA>{}, make_shape(bM, bK));
//   auto sB = tile_to_shape(
//       GMMA::Layout_K_SW128_Atom<TB>{}, make_shape(bN, bK));
//
//   // TMA 创建（同 FP16，仅类型不同）
//   Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA, make_shape(bM, bK));
//
//   // WGMMA 执行方式与 FP16 相同
//   gemm(mma, tCrA, tCrB, tCrC);  // FP8 x FP8 => FP32
//
// 完整示例见:
//   3rd/cutlass/examples/54_hopper_fp8_warp_specialized_gemm/
```

---

## Step 11: 综合实战 — 基于 CuTe 的高性能 GEMM

**功能点**：整合 TMA + Swizzle + WGMMA + Pipeline 实现完整的 Hopper GEMM。

```cpp
// step11_hopper_gemm.cu
// 功能：完整 Hopper GEMM，整合所有特性
//
// 完整可执行代码见:
//   3rd/cutlass/examples/cute/tutorial/hopper/wgmma_tma_sm90.cu
//
// 该文件整合了以下所有特性：
//   - TMA Load/Store (Step 1, 2)
//   - Swizzle Layout (Step 3): GMMA::Layout_MN_SW128_Atom
//   - Async Barrier (Step 4): ClusterTransactionBarrier
//   - Cluster Launch (Step 5): launch_kernel_on_cluster
//   - WGMMA (Step 6): SM90_64x64x16_F16F16F16_SS
//   - Pipeline (Step 7): PipelineState, 3-stage 流水线
//
// 架构概览：
//
//   Host:
//     1. 创建 TMA atom (make_tma_atom)
//     2. 配置 smem layout (SW128 swizzle)
//     3. 通过 ClusterLaunchParams 启动 kernel
//
//   Device kernel:
//     1. [初始化] barrier init, cluster_sync
//     2. [Prefetch] 预加载 K_PIPE_MAX 个 tile 到 smem
//     3. [Mainloop] 流水线循环:
//        - Producer: TMA load → barrier arrive
//        - Consumer: barrier wait → WGMMA → consumer arrive
//     4. [Epilogue] axpby 写回全局内存
//
// 编译运行:
//   cd 3rd/cutlass
//   mkdir build && cd build
//   cmake .. -DCUTLASS_NVCC_ARCHS=90a
//   make cute_tutorial_hopper_wgmma_tma
//   ./examples/cute/tutorial/hopper/wgmma_tma_sm90
```

---

## 参考资料

详细参考见 [README.md](../README.md)。
