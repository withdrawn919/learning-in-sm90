# Hopper SW spec in CUTLASS/CUTE

# cluster launch kernel

## \_\_cluster\_dims\_\_

\_\_cluster\_dims\_\_(2,1,1)表示cluster的维度是（2,1,1），每个cluster中包含两个block。

```c++
__global__ __cluster_dims__(2,1,1)
void kernel(float* ptr){}
```

启动kernel的代码如下，不用额外传入cluster的config参数。

```c++
kernel<<<grid, block>>>(ptr);
```

## cudaLaunchKernelEx

以<<<>>>的方式调用kernel的方式是cuda提供的语法糖，编译后都会去调用runtime中的launchkernel函数。

cuda\_runtime\_api.h中提供了两个launkernel的函数，第一个cudaLaunchKernel中只有gridDim、 blockDim、sharedMem等kernel config，并没有cluster 参数。Hopper架构以后nvidia在grid和block之间新增了cluster层级，新增了cudaLaunchKernelExC的函数来调用使用了cluster的kernel。

```cuda-cpp
#elif defined(__CUDART_API_PER_THREAD_DEFAULT_STREAM)
    // nvcc stubs reference the 'cudaLaunch'/'cudaLaunchKernel' identifier even if it was defined
    // to 'cudaLaunch_ptsz'/'cudaLaunchKernel_ptsz'. Redirect through a static inline function.
    #undef cudaLaunchKernel
    static __inline__ __host__ cudaError_t cudaLaunchKernel(const void *func, 
                                                            dim3 gridDim, dim3 blockDim, 
                                                            void **args, size_t sharedMem, 
                                                            cudaStream_t stream)
    {
        return cudaLaunchKernel_ptsz(func, gridDim, blockDim, args, sharedMem, stream);
    }
    #define cudaLaunchKernel __CUDART_API_PTSZ(cudaLaunchKernel)
    #undef cudaLaunchKernelExC
    static __inline__ __host__ cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config,
                                                               const void *func,
                                                                  void **args)
    {
        return cudaLaunchKernelExC_ptsz(config, func, args);
    }
    #define cudaLaunchKernelExC __CUDART_API_PTSZ(cudaLaunchKernelExC)
#endif

#if defined(__cplusplus)
}
```

cudaLaunchConfig\_t的struct如下，cluster的信息被写在cudaLaunchAttribute的结构体中。

```cuda-cpp
typedef __device_builtin__ struct cudaLaunchConfig_st {
    dim3 gridDim;               /**< Grid dimensions */
    dim3 blockDim;              /**< Block dimensions */
    size_t dynamicSmemBytes;    /**< Dynamic shared-memory size per thread block in bytes */
    cudaStream_t stream;        /**< Stream identifier */
    cudaLaunchAttribute *attrs; /**< List of attributes; nullable if ::cudaLaunchConfig_t::numAttrs == 0 */
    unsigned int numAttrs;      /**< Number of attributes populated in ::cudaLaunchConfig_t::attrs */
} cudaLaunchConfig_t;
```

想直接使用这个api比较麻烦，一般使用cutlass的封装。launch\_kernel\_on\_cluster将函数参数和kernel config正确初始化后调用cudaLaunchKernelEx。

```cuda-cpp
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(const ClusterLaunchParams& params,
                                                             void const* kernel_ptr, Args&& ... args)
```