// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

constexpr int kReadyQkv = 0;
constexpr int kReadyOutput = 1;

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

__global__ void dispatch_qkv_kernel(
    char* symmetric_data, std::size_t data_slot_bytes, const char* source,
    char* packed, const std::int64_t* route_indices,
    const std::int32_t* route_counts, const std::int32_t* peer_ranks,
    int peer_count, int batch_rows, int row_bytes, std::uint64_t* signals,
    std::uint64_t* epochs, int world_size, int local_rank, int layer_count,
    int layer_index) {
  const int peer_slot = blockIdx.x;
  if (peer_slot >= peer_count) {
    return;
  }
  const int count = route_counts[peer_slot];
  if (count <= 0) {
    return;
  }
  const int peer = peer_ranks[peer_slot];
  if (layer_index == 0 && threadIdx.x == 0) {
    epochs[peer] += 1;
  }
  __syncthreads();
  const std::uint64_t value =
      (epochs[peer] - 1) * static_cast<std::uint64_t>(layer_count) +
      static_cast<std::uint64_t>(layer_index) + 1;

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
    int world_size, int layer_count, int layer_index) {
  const int peer_slot = blockIdx.x;
  if (peer_slot >= peer_count) {
    return;
  }
  const int count = route_counts[peer_slot];
  if (count <= 0) {
    return;
  }
  const int peer = peer_ranks[peer_slot];
  const std::uint64_t value =
      (epochs[peer] - 1) * static_cast<std::uint64_t>(layer_count) +
      static_cast<std::uint64_t>(layer_index) + 1;
  if (threadIdx.x == 0) {
    nvshmem_uint64_wait_until(signals + kReadyOutput * world_size + peer,
                              NVSHMEM_CMP_GE, value);
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

int launch_status() { return static_cast<int>(cudaGetLastError()); }

}  // namespace

extern "C" int pap_nvshmem_device_bridge_version() { return 5; }

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
  void* arguments[] = {
      &destination,
      const_cast<void**>(&source),
      &num_bytes,
      &signal,
      const_cast<std::uint64_t**>(&abort_signal),
      const_cast<std::uint64_t**>(&epoch),
      &layer_count,
      &layer_index,
      &peer,
  };
  const cudaError_t status =
      cudaLaunchCooperativeKernel(reinterpret_cast<void*>(put_signal_kernel),
                                  dim3(1), dim3(256), arguments, 0, stream);
  return static_cast<int>(status);
}

extern "C" int pap_nvshmem_graph_dispatch_qkv(
    void* symmetric_data, std::size_t data_slot_bytes, const void* source,
    void* packed, const std::int64_t* route_indices,
    const std::int32_t* route_counts, const std::int32_t* peer_ranks,
    int peer_count, int batch_rows, int row_bytes, std::uint64_t* signals,
    std::uint64_t* epochs, int world_size, int local_rank, int layer_count,
    int layer_index, cudaStream_t stream) {
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
    int world_size, int layer_count, int layer_index, cudaStream_t stream) {
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
  };
  const cudaError_t status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(gather_output_kernel), dim3(peer_count),
      dim3(256), arguments, 0, stream);
  return static_cast<int>(status);
}
