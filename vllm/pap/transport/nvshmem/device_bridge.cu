// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cooperative_groups.h>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

constexpr int kReadyQkv = 0;
constexpr int kReadyOutput = 1;

struct DeviceGraphLauncher {
  cudaGraphExec_t child;
  cudaGraphExec_t launcher;
};

struct ResidentDispatchDescriptor {
  cudaGraphExec_t child;
  std::uint64_t generation;
  std::uint64_t stop;
};

struct ResidentDispatcher {
  ResidentDispatchDescriptor* host_descriptor;
  ResidentDispatchDescriptor* device_descriptor;
  cudaGraphExec_t launcher;
  cudaStream_t stream;
  std::uint64_t queued_until;
  int window_size;
};

__device__ __forceinline__ std::uint64_t globaltimer_ns() {
  std::uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ std::uint64_t generation(const std::uint64_t* epoch,
                                                    int layer_count,
                                                    int layer_index) {
  return (*epoch - 1) * static_cast<std::uint64_t>(layer_count) +
         static_cast<std::uint64_t>(layer_index) + 1;
}

__global__ void advance_epoch_kernel(std::uint64_t* epoch) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *epoch += 1;
  }
}

__global__ void wait_signal_kernel(std::uint64_t* signal,
                                   const std::uint64_t* epoch, int layer_count,
                                   int layer_index, int generation_delta) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const std::uint64_t expected =
        generation(epoch, layer_count, layer_index) + generation_delta;
    nvshmem_uint64_wait_until(signal, NVSHMEM_CMP_GE, expected);
  }
}

__global__ void put_signal_kernel(void* destination, const void* source,
                                  std::size_t num_bytes, std::uint64_t* signal,
                                  const std::uint64_t* abort_signal,
                                  const std::uint64_t* epoch, int layer_count,
                                  int layer_index, int peer) {
  if (*abort_signal != 0) {
    return;
  }
  const std::uint64_t value = generation(epoch, layer_count, layer_index);
  nvshmemx_putmem_signal_block(destination, source, num_bytes, signal, value,
                               NVSHMEM_SIGNAL_SET, peer);
}

__global__ void launch_device_graph_kernel(cudaGraphExec_t executable) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    cudaGraphLaunch(executable, cudaStreamGraphTailLaunch);
  }
}

__global__ void launch_resident_graph_kernel(
    const ResidentDispatchDescriptor* descriptor) {
  if (blockIdx.x == 0 && threadIdx.x == 0 && descriptor->stop == 0) {
    cudaGraphLaunch(descriptor->child, cudaStreamGraphTailLaunch);
  }
}

cudaError_t enqueue_resident_window(ResidentDispatcher* dispatcher,
                                    std::uint64_t first_generation) {
  const std::uint64_t last_generation =
      first_generation + dispatcher->window_size - 1;
  for (std::uint64_t generation = first_generation;
       generation <= last_generation; ++generation) {
    const CUresult wait_status =
        cuStreamWaitValue64(reinterpret_cast<CUstream>(dispatcher->stream),
                            reinterpret_cast<CUdeviceptr>(
                                &dispatcher->device_descriptor->generation),
                            generation, CU_STREAM_WAIT_VALUE_GEQ);
    if (wait_status != CUDA_SUCCESS) {
      return cudaErrorUnknown;
    }
    const cudaError_t launch_status =
        cudaGraphLaunch(dispatcher->launcher, dispatcher->stream);
    if (launch_status != cudaSuccess) {
      return launch_status;
    }
  }
  dispatcher->queued_until = last_generation;
  return cudaSuccess;
}

__global__ void dispatch_qkv_kernel(
    char* symmetric_data, std::size_t data_slot_bytes, const char* source,
    char* packed, const std::int64_t* route_indices,
    const std::int32_t* route_counts, const std::int32_t* peer_ranks,
    int peer_count, int batch_rows, int row_bytes, std::uint64_t* signals,
    std::uint64_t* epochs, int world_size, int local_rank, int layer_count,
    int layer_index, std::uint64_t* trace_start_ns,
    std::uint64_t* trace_step_ids, std::int32_t* trace_route_counts,
    std::uint64_t* trace_peer_epochs, std::uint64_t* trace_step_counter,
    std::uint64_t* trace_current_step, std::uint64_t* trace_host_completion,
    int trace_steps, int trace_layers) {
  const int peer_slot = blockIdx.x;
  if (peer_slot >= peer_count) {
    return;
  }
  const int count = route_counts[peer_slot];
  std::uint64_t trace_step = 0;
  if (trace_start_ns != nullptr && trace_steps > 0) {
    if (layer_index == 0) {
      if (peer_slot == 0 && threadIdx.x == 0) {
        *trace_current_step = atomicAdd(
            reinterpret_cast<unsigned long long*>(trace_step_counter), 1ULL);
      }
      cooperative_groups::this_grid().sync();
    }
    trace_step = *trace_current_step;
    if (layer_index == 0 && peer_slot == 0 && threadIdx.x == 0 &&
        trace_host_completion != nullptr) {
      const int slot = static_cast<int>(trace_step % trace_steps);
      trace_host_completion[slot] = trace_step * 2 + 1;
      __threadfence_system();
    }
    if (layer_index == 0 && threadIdx.x == 0) {
      const int slot = static_cast<int>(trace_step % trace_steps);
      const int peer = peer_ranks[peer_slot];
      const std::size_t step_offset =
          static_cast<std::size_t>(slot) * world_size + peer;
      trace_step_ids[step_offset] = trace_step;
      trace_route_counts[step_offset] = count;
    }
  }
  if (count <= 0) {
    return;
  }
  const int peer = peer_ranks[peer_slot];
  if (layer_index == 0 && threadIdx.x == 0) {
    epochs[peer] += 1;
  }
  __syncthreads();
  if (layer_index == 0 && threadIdx.x == 0 && trace_peer_epochs != nullptr) {
    const int slot = static_cast<int>(trace_step % trace_steps);
    const std::size_t step_offset =
        static_cast<std::size_t>(slot) * world_size + peer;
    trace_peer_epochs[step_offset] = epochs[peer] - 1;
  }
  const std::uint64_t value =
      (epochs[peer] - 1) * static_cast<std::uint64_t>(layer_count) +
      static_cast<std::uint64_t>(layer_index) + 1;
  if (threadIdx.x == 0 && trace_start_ns != nullptr && trace_steps > 0 &&
      layer_index < trace_layers) {
    const int slot = static_cast<int>(trace_step % trace_steps);
    const std::size_t layer_offset =
        (static_cast<std::size_t>(slot) * trace_layers + layer_index) *
        world_size;
    trace_start_ns[layer_offset + peer] = globaltimer_ns();
  }

  char* peer_packed =
      packed + static_cast<std::size_t>(peer_slot) * batch_rows * row_bytes;
  const int payload_bytes = count * row_bytes;
  for (int offset = threadIdx.x; offset < payload_bytes; offset += blockDim.x) {
    const int route_row = offset / row_bytes;
    const int byte_in_row = offset - route_row * row_bytes;
    const std::int64_t source_row =
        route_indices[peer_slot * batch_rows + route_row];
    peer_packed[offset] = source[source_row * row_bytes + byte_in_row];
  }
  __syncthreads();
  nvshmemx_putmem_signal_block(
      symmetric_data + static_cast<std::size_t>(local_rank) * data_slot_bytes,
      peer_packed, payload_bytes, signals + kReadyQkv * world_size + local_rank,
      value, NVSHMEM_SIGNAL_SET, peer);
}

__global__ void gather_output_kernel(
    const char* symmetric_data, std::size_t data_slot_bytes, char* output,
    const std::int64_t* route_indices, const std::int32_t* route_counts,
    const std::int32_t* peer_ranks, int peer_count, int batch_rows,
    int row_bytes, std::uint64_t* signals, const std::uint64_t* epochs,
    int world_size, int layer_count, int layer_index,
    std::uint64_t* trace_end_ns, const std::uint64_t* trace_current_step,
    int trace_steps, int trace_layers) {
  const int peer_slot = blockIdx.x;
  if (peer_slot >= peer_count) {
    return;
  }
  const int count = route_counts[peer_slot];
  if (count <= 0) {
    return;
  }
  const int peer = peer_ranks[peer_slot];
  if (threadIdx.x == 0) {
    const std::uint64_t value =
        (epochs[peer] - 1) * static_cast<std::uint64_t>(layer_count) +
        static_cast<std::uint64_t>(layer_index) + 1;
    nvshmem_uint64_wait_until(signals + kReadyOutput * world_size + peer,
                              NVSHMEM_CMP_GE, value);
    if (trace_end_ns != nullptr && trace_steps > 0 &&
        layer_index < trace_layers) {
      const int slot = static_cast<int>(*trace_current_step % trace_steps);
      const std::size_t layer_offset =
          (static_cast<std::size_t>(slot) * trace_layers + layer_index) *
          world_size;
      trace_end_ns[layer_offset + peer] = globaltimer_ns();
    }
  }
  __syncthreads();

  const char* peer_output =
      symmetric_data + static_cast<std::size_t>(peer) * data_slot_bytes;
  const int payload_bytes = count * row_bytes;
  for (int offset = threadIdx.x; offset < payload_bytes; offset += blockDim.x) {
    const int route_row = offset / row_bytes;
    const int byte_in_row = offset - route_row * row_bytes;
    const std::int64_t output_row =
        route_indices[peer_slot * batch_rows + route_row];
    output[output_row * row_bytes + byte_in_row] = peer_output[offset];
  }
}

__global__ void projection_dispatch_done_kernel(
    const std::uint64_t* current_step, std::uint64_t* dispatch_done_ns,
    int trace_steps, int trace_layers, int layer_index) {
  if (threadIdx.x == 0) {
    const std::uint64_t step = *current_step;
    const int slot = static_cast<int>(step % trace_steps);
    dispatch_done_ns[static_cast<std::size_t>(slot) * trace_layers +
                     layer_index] = globaltimer_ns();
  }
}

__global__ void projection_gather_done_kernel(
    const std::uint64_t* current_step, const std::uint64_t* start_ns,
    const std::uint64_t* end_ns, const std::uint64_t* step_ids,
    const std::int32_t* route_counts, const std::uint64_t* peer_epochs,
    const std::uint64_t* dispatch_done_ns, std::uint64_t* gather_done_ns,
    std::uint64_t* host_start_ns, std::uint64_t* host_end_ns,
    std::uint64_t* host_step_ids, std::int32_t* host_route_counts,
    std::uint64_t* host_peer_epochs, std::uint64_t* host_dispatch_done_ns,
    std::uint64_t* host_gather_done_ns, std::uint64_t* host_completion,
    int trace_steps, int trace_layers, int world_size, int layer_index) {
  const std::uint64_t step = *current_step;
  const int slot = static_cast<int>(step % trace_steps);
  const std::size_t layer_offset =
      static_cast<std::size_t>(slot) * trace_layers + layer_index;
  if (threadIdx.x == 0) {
    gather_done_ns[layer_offset] = globaltimer_ns();
  }
  __syncthreads();
  if (layer_index + 1 != trace_layers) {
    return;
  }
  const std::size_t peer_layer_values =
      static_cast<std::size_t>(trace_layers) * world_size;
  const std::size_t peer_layer_offset =
      static_cast<std::size_t>(slot) * peer_layer_values;
  for (std::size_t index = threadIdx.x; index < peer_layer_values;
       index += blockDim.x) {
    host_start_ns[peer_layer_offset + index] =
        start_ns[peer_layer_offset + index];
    host_end_ns[peer_layer_offset + index] = end_ns[peer_layer_offset + index];
  }
  const std::size_t step_offset = static_cast<std::size_t>(slot) * world_size;
  for (int index = threadIdx.x; index < world_size; index += blockDim.x) {
    host_step_ids[step_offset + index] = step_ids[step_offset + index];
    host_route_counts[step_offset + index] = route_counts[step_offset + index];
    host_peer_epochs[step_offset + index] = peer_epochs[step_offset + index];
  }
  const std::size_t scalar_layer_offset =
      static_cast<std::size_t>(slot) * trace_layers;
  for (int index = threadIdx.x; index < trace_layers; index += blockDim.x) {
    host_dispatch_done_ns[scalar_layer_offset + index] =
        dispatch_done_ns[scalar_layer_offset + index];
    host_gather_done_ns[scalar_layer_offset + index] =
        gather_done_ns[scalar_layer_offset + index];
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    host_completion[slot] = step * 2 + 2;
    __threadfence_system();
  }
}

__global__ void attention_trace_marker_kernel(
    const std::uint64_t* epoch, std::uint64_t* replay_start_ns,
    std::uint64_t* step_start_ns, std::uint64_t* start_ns,
    std::uint64_t* end_ns, std::uint64_t* step_ids,
    std::uint64_t* host_replay_start_ns, std::uint64_t* host_step_start_ns,
    std::uint64_t* host_start_ns, std::uint64_t* host_end_ns,
    std::uint64_t* host_step_ids, std::uint64_t* host_completion,
    int trace_steps, int trace_layers, int layer_index, int marker_kind) {
  const std::uint64_t step = marker_kind == 3 ? *epoch : *epoch - 1;
  const int slot = static_cast<int>(step % trace_steps);
  const std::size_t offset =
      static_cast<std::size_t>(slot) * trace_layers + layer_index;
  if (threadIdx.x == 0) {
    if (marker_kind == 3) {
      replay_start_ns[slot] = globaltimer_ns();
    } else if (marker_kind == 0) {
      host_completion[slot] = step * 2 + 1;
      __threadfence_system();
      step_ids[slot] = step;
      step_start_ns[slot] = globaltimer_ns();
    } else if (marker_kind == 1) {
      start_ns[offset] = globaltimer_ns();
    } else {
      end_ns[offset] = globaltimer_ns();
    }
  }
  __syncthreads();
  if (marker_kind != 2 || layer_index + 1 != trace_layers) {
    return;
  }
  const std::size_t step_offset = static_cast<std::size_t>(slot) * trace_layers;
  for (int index = threadIdx.x; index < trace_layers; index += blockDim.x) {
    host_start_ns[step_offset + index] = start_ns[step_offset + index];
    host_end_ns[step_offset + index] = end_ns[step_offset + index];
  }
  if (threadIdx.x == 0) {
    host_replay_start_ns[slot] = replay_start_ns[slot];
    host_step_start_ns[slot] = step_start_ns[slot];
    host_step_ids[slot] = step_ids[slot];
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    host_completion[slot] = step * 2 + 2;
    __threadfence_system();
  }
}

int launch_status() { return static_cast<int>(cudaGetLastError()); }

}  // namespace

extern "C" int pap_nvshmem_device_bridge_version() { return 10; }

extern "C" int pap_cuda_graph_probe_device_launch(
    void* graph_handle, cudaStream_t stream, int* result_out,
    int* node_type_out, char* node_name_out, std::size_t node_name_bytes) {
  if (graph_handle == nullptr || stream == nullptr || result_out == nullptr ||
      node_type_out == nullptr || node_name_out == nullptr ||
      node_name_bytes == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *result_out = static_cast<int>(cudaGraphInstantiateError);
  *node_type_out = -1;
  node_name_out[0] = '\0';
  cudaGraph_t graph = reinterpret_cast<cudaGraph_t>(graph_handle);
  cudaGraph_t clone = nullptr;
  cudaGraphExec_t executable = nullptr;
  cudaError_t status = cudaGraphClone(&clone, graph);
  if (status == cudaSuccess) {
    cudaGraphInstantiateParams parameters{};
    parameters.flags = cudaGraphInstantiateFlagDeviceLaunch;
    status = cudaGraphInstantiateWithParams(&executable, clone, &parameters);
    *result_out = static_cast<int>(parameters.result_out);
    if (parameters.errNode_out != nullptr) {
      cudaGraphNodeType node_type;
      if (cudaGraphNodeGetType(parameters.errNode_out, &node_type) ==
          cudaSuccess) {
        *node_type_out = static_cast<int>(node_type);
        if (node_type == cudaGraphNodeTypeKernel) {
          cudaKernelNodeParams kernel_parameters{};
          const char* function_name = nullptr;
          if (cudaGraphKernelNodeGetParams(parameters.errNode_out,
                                           &kernel_parameters) == cudaSuccess &&
              cudaFuncGetName(&function_name, kernel_parameters.func) ==
                  cudaSuccess &&
              function_name != nullptr) {
            std::strncpy(node_name_out, function_name, node_name_bytes - 1);
            node_name_out[node_name_bytes - 1] = '\0';
          }
        }
      }
    }
  }
  if (status == cudaSuccess) {
    status = cudaGraphUpload(executable, stream);
  }
  if (status == cudaSuccess) {
    status = cudaStreamSynchronize(stream);
  }
  if (executable != nullptr) {
    cudaGraphExecDestroy(executable);
  }
  if (clone != nullptr) {
    cudaGraphDestroy(clone);
  }
  return static_cast<int>(status);
}

extern "C" int pap_cuda_graph_create_resident_dispatcher(
    cudaStream_t stream, int window_size, void** dispatcher_out) {
  if (stream == nullptr || window_size <= 0 || dispatcher_out == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *dispatcher_out = nullptr;
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return static_cast<int>(cudaErrorNoDevice);
  }
  int stream_memory_operations = 0;
  if (cuDeviceGetAttribute(&stream_memory_operations,
                           CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS,
                           device) != CUDA_SUCCESS ||
      stream_memory_operations == 0) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  ResidentDispatchDescriptor* host_descriptor = nullptr;
  cudaError_t status =
      cudaHostAlloc(&host_descriptor, sizeof(ResidentDispatchDescriptor),
                    cudaHostAllocMapped | cudaHostAllocPortable);
  ResidentDispatchDescriptor* device_descriptor = nullptr;
  if (status == cudaSuccess) {
    std::memset(host_descriptor, 0, sizeof(*host_descriptor));
    status = cudaHostGetDevicePointer(&device_descriptor, host_descriptor, 0);
  }
  cudaGraph_t launcher_graph = nullptr;
  cudaGraphExec_t launcher = nullptr;
  if (status == cudaSuccess) {
    status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  }
  if (status == cudaSuccess) {
    launch_resident_graph_kernel<<<1, 1, 0, stream>>>(device_descriptor);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess) {
    status = cudaStreamEndCapture(stream, &launcher_graph);
  }
  if (status == cudaSuccess) {
    status = cudaGraphInstantiate(&launcher, launcher_graph, 0);
  }
  if (status == cudaSuccess) {
    status = cudaGraphUpload(launcher, stream);
  }
  if (status == cudaSuccess) {
    status = cudaStreamSynchronize(stream);
  }
  if (launcher_graph != nullptr) {
    cudaGraphDestroy(launcher_graph);
  }
  if (status != cudaSuccess) {
    if (launcher != nullptr) {
      cudaGraphExecDestroy(launcher);
    }
    if (host_descriptor != nullptr) {
      cudaFreeHost(host_descriptor);
    }
    return static_cast<int>(status);
  }
  auto* dispatcher = new ResidentDispatcher{
      host_descriptor, device_descriptor, launcher, stream, 0, window_size};
  *dispatcher_out = dispatcher;
  return static_cast<int>(cudaSuccess);
}

extern "C" int pap_cuda_graph_create_device_launch(
    void* graph_handle, cudaStream_t stream, void** executable_out,
    int* result_out, int* node_type_out, char* node_name_out,
    std::size_t node_name_bytes) {
  if (graph_handle == nullptr || stream == nullptr ||
      executable_out == nullptr || result_out == nullptr ||
      node_type_out == nullptr || node_name_out == nullptr ||
      node_name_bytes == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *executable_out = nullptr;
  *result_out = static_cast<int>(cudaGraphInstantiateError);
  *node_type_out = -1;
  node_name_out[0] = '\0';
  cudaGraph_t graph = reinterpret_cast<cudaGraph_t>(graph_handle);
  cudaGraph_t clone = nullptr;
  cudaGraphExec_t child = nullptr;
  cudaGraph_t launcher_graph = nullptr;
  cudaGraphExec_t launcher = nullptr;
  cudaError_t status = cudaGraphClone(&clone, graph);
  if (status == cudaSuccess) {
    cudaGraphInstantiateParams parameters{};
    parameters.flags = cudaGraphInstantiateFlagDeviceLaunch;
    status = cudaGraphInstantiateWithParams(&child, clone, &parameters);
    *result_out = static_cast<int>(parameters.result_out);
    if (parameters.errNode_out != nullptr) {
      cudaGraphNodeType node_type;
      if (cudaGraphNodeGetType(parameters.errNode_out, &node_type) ==
          cudaSuccess) {
        *node_type_out = static_cast<int>(node_type);
        if (node_type == cudaGraphNodeTypeKernel) {
          cudaKernelNodeParams kernel_parameters{};
          const char* function_name = nullptr;
          if (cudaGraphKernelNodeGetParams(parameters.errNode_out,
                                           &kernel_parameters) == cudaSuccess &&
              cudaFuncGetName(&function_name, kernel_parameters.func) ==
                  cudaSuccess &&
              function_name != nullptr) {
            std::strncpy(node_name_out, function_name, node_name_bytes - 1);
            node_name_out[node_name_bytes - 1] = '\0';
          }
        }
      }
    }
  }
  if (status == cudaSuccess) {
    status = cudaGraphUpload(child, stream);
  }
  if (status == cudaSuccess) {
    status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  }
  if (status == cudaSuccess) {
    launch_device_graph_kernel<<<1, 1, 0, stream>>>(child);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess) {
    status = cudaStreamEndCapture(stream, &launcher_graph);
  }
  if (status == cudaSuccess) {
    status = cudaGraphInstantiate(&launcher, launcher_graph, 0);
  }
  if (status == cudaSuccess) {
    status = cudaGraphUpload(launcher, stream);
  }
  if (status == cudaSuccess) {
    status = cudaStreamSynchronize(stream);
  }
  if (clone != nullptr) {
    cudaGraphDestroy(clone);
  }
  if (launcher_graph != nullptr) {
    cudaGraphDestroy(launcher_graph);
  }
  if (status != cudaSuccess) {
    if (launcher != nullptr) {
      cudaGraphExecDestroy(launcher);
    }
    if (child != nullptr) {
      cudaGraphExecDestroy(child);
    }
    return static_cast<int>(status);
  }
  *executable_out = new DeviceGraphLauncher{child, launcher};
  return static_cast<int>(status);
}

extern "C" int pap_cuda_graph_resident_run(void* dispatcher_handle,
                                           void* executable_handle,
                                           std::uint64_t generation) {
  if (dispatcher_handle == nullptr || executable_handle == nullptr ||
      generation == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  auto* dispatcher = static_cast<ResidentDispatcher*>(dispatcher_handle);
  auto* executable = static_cast<DeviceGraphLauncher*>(executable_handle);
  if (generation > dispatcher->queued_until) {
    const cudaError_t status =
        enqueue_resident_window(dispatcher, dispatcher->queued_until + 1);
    if (status != cudaSuccess) {
      return static_cast<int>(status);
    }
  }
  __atomic_store_n(&dispatcher->host_descriptor->child, executable->child,
                   __ATOMIC_RELAXED);
  __atomic_store_n(&dispatcher->host_descriptor->stop, 0, __ATOMIC_RELAXED);
  __atomic_store_n(&dispatcher->host_descriptor->generation, generation,
                   __ATOMIC_RELEASE);
  return static_cast<int>(cudaStreamSynchronize(dispatcher->stream));
}

extern "C" int pap_cuda_graph_destroy_resident_dispatcher(
    void* dispatcher_handle) {
  if (dispatcher_handle == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  auto* dispatcher = static_cast<ResidentDispatcher*>(dispatcher_handle);
  __atomic_store_n(&dispatcher->host_descriptor->stop, 1, __ATOMIC_RELAXED);
  __atomic_store_n(&dispatcher->host_descriptor->generation,
                   dispatcher->queued_until, __ATOMIC_RELEASE);
  cudaError_t status = cudaStreamSynchronize(dispatcher->stream);
  const cudaError_t destroy_status = cudaGraphExecDestroy(dispatcher->launcher);
  const cudaError_t free_status = cudaFreeHost(dispatcher->host_descriptor);
  delete dispatcher;
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (destroy_status != cudaSuccess) {
    return static_cast<int>(destroy_status);
  }
  return static_cast<int>(free_status);
}

extern "C" int pap_cuda_graph_launch_from_device(void* executable_handle,
                                                 cudaStream_t stream) {
  if (executable_handle == nullptr || stream == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  auto* executable = static_cast<DeviceGraphLauncher*>(executable_handle);
  return static_cast<int>(cudaGraphLaunch(executable->launcher, stream));
}

extern "C" int pap_cuda_graph_destroy_device_launch(void* executable_handle) {
  if (executable_handle == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  auto* executable = static_cast<DeviceGraphLauncher*>(executable_handle);
  cudaError_t status = cudaGraphExecDestroy(executable->launcher);
  const cudaError_t child_status = cudaGraphExecDestroy(executable->child);
  delete executable;
  return static_cast<int>(status != cudaSuccess ? status : child_status);
}

extern "C" void* pap_cuda_host_get_device_pointer(void* host_pointer) {
  void* device_pointer = nullptr;
  if (cudaHostGetDevicePointer(&device_pointer, host_pointer, 0) !=
      cudaSuccess) {
    return nullptr;
  }
  return device_pointer;
}

extern "C" int pap_nvshmem_device_bridge_get_unique_id(void* output,
                                                       std::size_t num_bytes) {
  if (output == nullptr || num_bytes != sizeof(nvshmemx_uniqueid_t)) {
    return 22;
  }
  nvshmemx_uniqueid_t unique_id = NVSHMEMX_UNIQUEID_INITIALIZER;
  const int status = nvshmemx_get_uniqueid(&unique_id);
  if (status == 0) {
    std::memcpy(output, &unique_id, sizeof(unique_id));
  }
  return status;
}

extern "C" int pap_nvshmem_device_bridge_init_uid(const void* unique_id_bytes,
                                                  std::size_t num_bytes,
                                                  int rank, int world_size) {
  if (unique_id_bytes == nullptr || num_bytes != sizeof(nvshmemx_uniqueid_t)) {
    return 22;
  }
  nvshmemx_uniqueid_t unique_id = NVSHMEMX_UNIQUEID_INITIALIZER;
  std::memcpy(&unique_id, unique_id_bytes, sizeof(unique_id));
  nvshmemx_init_attr_t attributes = NVSHMEMX_INIT_ATTR_INITIALIZER;
  int status = nvshmemx_set_attr_uniqueid_args(rank, world_size, &unique_id,
                                               &attributes);
  if (status != 0) {
    return status;
  }
  return nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attributes);
}

extern "C" void pap_nvshmem_device_bridge_finalize() { nvshmem_finalize(); }

extern "C" int pap_nvshmem_device_bridge_my_pe() { return nvshmem_my_pe(); }

extern "C" int pap_nvshmem_device_bridge_n_pes() { return nvshmem_n_pes(); }

extern "C" void* pap_nvshmem_device_bridge_malloc(std::size_t num_bytes) {
  return nvshmem_malloc(num_bytes);
}

extern "C" void pap_nvshmem_device_bridge_free(void* pointer) {
  nvshmem_free(pointer);
}

extern "C" void pap_nvshmem_device_bridge_barrier() { nvshmem_barrier_all(); }

extern "C" void pap_nvshmem_device_bridge_put_signal_on_stream(
    void* destination, const void* source, std::size_t num_bytes,
    std::uint64_t* signal, std::uint64_t generation, int signal_operation,
    int peer, cudaStream_t stream) {
  nvshmemx_putmem_signal_on_stream(destination, source, num_bytes, signal,
                                   generation, signal_operation, peer, stream);
}

extern "C" void pap_nvshmem_device_bridge_signal_on_stream(
    std::uint64_t* signal, std::uint64_t generation, int signal_operation,
    int peer, cudaStream_t stream) {
  nvshmemx_signal_op_on_stream(signal, generation, signal_operation, peer,
                               stream);
}

extern "C" int pap_nvshmem_graph_advance_epoch(std::uint64_t* epoch,
                                               cudaStream_t stream) {
  advance_epoch_kernel<<<1, 1, 0, stream>>>(epoch);
  return launch_status();
}

extern "C" int pap_nvshmem_graph_wait_signal(std::uint64_t* signal,
                                             const std::uint64_t* epoch,
                                             int layer_count, int layer_index,
                                             int generation_delta,
                                             cudaStream_t stream) {
  wait_signal_kernel<<<1, 32, 0, stream>>>(signal, epoch, layer_count,
                                           layer_index, generation_delta);
  return launch_status();
}

extern "C" int pap_nvshmem_graph_put_signal(
    void* destination, const void* source, std::size_t num_bytes,
    std::uint64_t* signal, const std::uint64_t* abort_signal,
    const std::uint64_t* epoch, int layer_count, int layer_index, int peer,
    cudaStream_t stream) {
  put_signal_kernel<<<1, 256, 0, stream>>>(destination, source, num_bytes,
                                           signal, abort_signal, epoch,
                                           layer_count, layer_index, peer);
  return launch_status();
}

extern "C" int pap_nvshmem_graph_dispatch_qkv(
    void* symmetric_data, std::size_t data_slot_bytes, const void* source,
    void* packed, const std::int64_t* route_indices,
    const std::int32_t* route_counts, const std::int32_t* peer_ranks,
    int peer_count, int batch_rows, int row_bytes, std::uint64_t* signals,
    std::uint64_t* epochs, int world_size, int local_rank, int layer_count,
    int layer_index, std::uint64_t* trace_start_ns,
    std::uint64_t* trace_step_ids, std::int32_t* trace_route_counts,
    std::uint64_t* trace_peer_epochs, std::uint64_t* trace_step_counter,
    std::uint64_t* trace_current_step, std::uint64_t* trace_host_completion,
    int trace_steps, int trace_layers, cudaStream_t stream) {
  void* arguments[] = {
      &symmetric_data,
      &data_slot_bytes,
      const_cast<void**>(&source),
      &packed,
      const_cast<std::int64_t**>(&route_indices),
      const_cast<std::int32_t**>(&route_counts),
      const_cast<std::int32_t**>(&peer_ranks),
      &peer_count,
      &batch_rows,
      &row_bytes,
      &signals,
      &epochs,
      &world_size,
      &local_rank,
      &layer_count,
      &layer_index,
      &trace_start_ns,
      &trace_step_ids,
      &trace_route_counts,
      &trace_peer_epochs,
      &trace_step_counter,
      &trace_current_step,
      &trace_host_completion,
      &trace_steps,
      &trace_layers,
  };
  const cudaError_t status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(dispatch_qkv_kernel), dim3(peer_count), dim3(256),
      arguments, 0, stream);
  return static_cast<int>(status);
}

extern "C" int pap_nvshmem_graph_gather_output(
    const void* symmetric_data, std::size_t data_slot_bytes, void* output,
    const std::int64_t* route_indices, const std::int32_t* route_counts,
    const std::int32_t* peer_ranks, int peer_count, int batch_rows,
    int row_bytes, std::uint64_t* signals, const std::uint64_t* epochs,
    int world_size, int layer_count, int layer_index,
    std::uint64_t* trace_end_ns, const std::uint64_t* trace_current_step,
    int trace_steps, int trace_layers, cudaStream_t stream) {
  void* arguments[] = {
      const_cast<void**>(&symmetric_data),
      &data_slot_bytes,
      &output,
      const_cast<std::int64_t**>(&route_indices),
      const_cast<std::int32_t**>(&route_counts),
      const_cast<std::int32_t**>(&peer_ranks),
      &peer_count,
      &batch_rows,
      &row_bytes,
      &signals,
      const_cast<std::uint64_t**>(&epochs),
      &world_size,
      &layer_count,
      &layer_index,
      &trace_end_ns,
      const_cast<std::uint64_t**>(&trace_current_step),
      &trace_steps,
      &trace_layers,
  };
  const cudaError_t status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(gather_output_kernel), dim3(peer_count),
      dim3(256), arguments, 0, stream);
  return static_cast<int>(status);
}

extern "C" int pap_trace_projection_dispatch_done(
    const std::uint64_t* current_step, std::uint64_t* dispatch_done_ns,
    int trace_steps, int trace_layers, int layer_index, cudaStream_t stream) {
  projection_dispatch_done_kernel<<<1, 1, 0, stream>>>(
      current_step, dispatch_done_ns, trace_steps, trace_layers, layer_index);
  return launch_status();
}

extern "C" int pap_trace_projection_gather_done(
    const std::uint64_t* current_step, const std::uint64_t* start_ns,
    const std::uint64_t* end_ns, const std::uint64_t* step_ids,
    const std::int32_t* route_counts, const std::uint64_t* peer_epochs,
    const std::uint64_t* dispatch_done_ns, std::uint64_t* gather_done_ns,
    std::uint64_t* host_start_ns, std::uint64_t* host_end_ns,
    std::uint64_t* host_step_ids, std::int32_t* host_route_counts,
    std::uint64_t* host_peer_epochs, std::uint64_t* host_dispatch_done_ns,
    std::uint64_t* host_gather_done_ns, std::uint64_t* host_completion,
    int trace_steps, int trace_layers, int world_size, int layer_index,
    cudaStream_t stream) {
  projection_gather_done_kernel<<<1, 256, 0, stream>>>(
      current_step, start_ns, end_ns, step_ids, route_counts, peer_epochs,
      dispatch_done_ns, gather_done_ns, host_start_ns, host_end_ns,
      host_step_ids, host_route_counts, host_peer_epochs, host_dispatch_done_ns,
      host_gather_done_ns, host_completion, trace_steps, trace_layers,
      world_size, layer_index);
  return launch_status();
}

extern "C" int pap_trace_attention_marker(
    const std::uint64_t* epoch, std::uint64_t* replay_start_ns,
    std::uint64_t* step_start_ns, std::uint64_t* start_ns,
    std::uint64_t* end_ns, std::uint64_t* step_ids,
    std::uint64_t* host_replay_start_ns, std::uint64_t* host_step_start_ns,
    std::uint64_t* host_start_ns, std::uint64_t* host_end_ns,
    std::uint64_t* host_step_ids, std::uint64_t* host_completion,
    int trace_steps, int trace_layers, int layer_index, int marker_kind,
    cudaStream_t stream) {
  attention_trace_marker_kernel<<<1, 256, 0, stream>>>(
      epoch, replay_start_ns, step_start_ns, start_ns, end_ns, step_ids,
      host_replay_start_ns, host_step_start_ns, host_start_ns, host_end_ns,
      host_step_ids, host_completion, trace_steps, trace_layers, layer_index,
      marker_kind);
  return launch_status();
}
