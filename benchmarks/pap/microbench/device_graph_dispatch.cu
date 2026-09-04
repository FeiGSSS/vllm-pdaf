// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#define CUDA_CHECK(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = (call);                                        \
    if (status_ != cudaSuccess) {                                        \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(status_));                         \
      nvshmem_global_exit(1);                                            \
    }                                                                    \
  } while (0)

#define CU_CHECK(call)                                                         \
  do {                                                                         \
    CUresult status_ = (call);                                                 \
    if (status_ != CUDA_SUCCESS) {                                             \
      const char* name_ = nullptr;                                             \
      const char* message_ = nullptr;                                          \
      cuGetErrorName(status_, &name_);                                         \
      cuGetErrorString(status_, &message_);                                    \
      std::fprintf(stderr, "%s:%d driver error: %s: %s\n", __FILE__, __LINE__, \
                   name_ != nullptr ? name_ : "unknown",                       \
                   message_ != nullptr ? message_ : "unknown");                \
      nvshmem_global_exit(1);                                                  \
    }                                                                          \
  } while (0)

namespace {

constexpr std::uint32_t kStop = 1;

struct Options {
  int warmup = 100;
  int iterations = 1000;
  int child_work_ns = 5000;
  int requests = 12;
  int publish_gap_ns = 0;
  bool stream_wait = false;
};

struct StepDescriptor {
  std::uint64_t generation;
  std::uint32_t plan_slot;
  std::uint32_t flags;
};

struct DispatchState {
  std::uint64_t expected_generation;
  std::uint64_t base_generation;
  std::uint64_t current_generation;
  std::uint32_t current_index;
  std::uint32_t errors;
};

int parse_positive(const char* name, const char* value) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0 || parsed > INT32_MAX) {
    std::fprintf(stderr, "%s must be a positive integer\n", name);
    std::exit(2);
  }
  return static_cast<int>(parsed);
}

Options parse_args(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc;) {
    if (std::strcmp(argv[index], "--stream-wait") == 0) {
      options.stream_wait = true;
      ++index;
      continue;
    }
    if (index + 1 >= argc) {
      std::fprintf(stderr, "missing value for %s\n", argv[index]);
      std::exit(2);
    }
    if (std::strcmp(argv[index], "--warmup") == 0) {
      options.warmup = parse_positive("--warmup", argv[index + 1]);
    } else if (std::strcmp(argv[index], "--iterations") == 0) {
      options.iterations = parse_positive("--iterations", argv[index + 1]);
    } else if (std::strcmp(argv[index], "--child-work-ns") == 0) {
      options.child_work_ns =
          parse_positive("--child-work-ns", argv[index + 1]);
    } else if (std::strcmp(argv[index], "--requests") == 0) {
      options.requests = parse_positive("--requests", argv[index + 1]);
    } else if (std::strcmp(argv[index], "--publish-gap-ns") == 0) {
      options.publish_gap_ns =
          parse_positive("--publish-gap-ns", argv[index + 1]);
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", argv[index]);
      std::exit(2);
    }
    index += 2;
  }
  return options;
}

__device__ __forceinline__ std::uint64_t globaltimer_ns() {
  std::uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__global__ void child_kernel(DispatchState* state, std::uint64_t* signals,
                             std::uint64_t* child_start_ns,
                             std::uint64_t* child_end_ns, int* results,
                             int* seq_lens, const int* block_table,
                             int* graph_slots, int* pat_deltas,
                             int blocks_per_request, int request_count,
                             int plan_slot, int work_ns) {
  const std::uint32_t index = state->current_index;
  const std::uint64_t generation = state->current_generation;
  std::uint64_t start = 0;
  if (threadIdx.x == 0) {
    start = globaltimer_ns();
    child_start_ns[index] = start;
  }
  for (int row = threadIdx.x; row < request_count; row += blockDim.x) {
    const int prior_seq_len = seq_lens[row];
    const int logical_block = prior_seq_len / 16;
    const int offset = prior_seq_len % 16;
    const int physical_block =
        block_table[row * blocks_per_request + logical_block];
    graph_slots[row] = physical_block * 16 + offset;
    seq_lens[row] = prior_seq_len + 1;
    pat_deltas[row] += 1;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    while (globaltimer_ns() - start < static_cast<std::uint64_t>(work_ns)) {
      __nanosleep(64);
    }
    results[index] = plan_slot;
    child_end_ns[index] = globaltimer_ns();
    __threadfence_system();
    nvshmemx_signal_op(signals + 1, generation, NVSHMEM_SIGNAL_SET, 0);
  }
}

__global__ void dispatcher_kernel(DispatchState* state,
                                  const StepDescriptor* descriptor,
                                  std::uint64_t* signals,
                                  std::uint64_t* observed_ns,
                                  cudaGraphExec_t plan_zero,
                                  cudaGraphExec_t plan_one) {
  if (threadIdx.x != 0) {
    return;
  }
  const std::uint64_t generation = state->expected_generation;
  nvshmem_uint64_wait_until(signals, NVSHMEM_CMP_GE, generation);
  const StepDescriptor step = *descriptor;
  if (step.generation != generation) {
    state->errors += 1;
  }
  if ((step.flags & kStop) != 0) {
    nvshmemx_signal_op(signals + 1, generation, NVSHMEM_SIGNAL_SET, 0);
    return;
  }
  const std::uint32_t index =
      static_cast<std::uint32_t>(generation - state->base_generation);
  state->current_generation = generation;
  state->current_index = index;
  state->expected_generation = generation + 1;
  observed_ns[index] = globaltimer_ns();
  __threadfence_system();
  const cudaGraphExec_t child = step.plan_slot == 0 ? plan_zero : plan_one;
  cudaGraphLaunch(child, cudaStreamGraphTailLaunch);
  cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
}

__global__ void dispatch_once_kernel(DispatchState* state,
                                     const StepDescriptor* descriptor,
                                     std::uint64_t* signals,
                                     std::uint64_t* observed_ns,
                                     cudaGraphExec_t plan_zero,
                                     cudaGraphExec_t plan_one) {
  if (threadIdx.x != 0) {
    return;
  }
  const std::uint64_t generation = state->expected_generation;
  const StepDescriptor step = *descriptor;
  if (step.generation != generation) {
    state->errors += 1;
  }
  if ((step.flags & kStop) != 0) {
    nvshmemx_signal_op(signals + 1, generation, NVSHMEM_SIGNAL_SET, 0);
    return;
  }
  const std::uint32_t index =
      static_cast<std::uint32_t>(generation - state->base_generation);
  state->current_generation = generation;
  state->current_index = index;
  state->expected_generation = generation + 1;
  observed_ns[index] = globaltimer_ns();
  __threadfence_system();
  const cudaGraphExec_t child = step.plan_slot == 0 ? plan_zero : plan_one;
  cudaGraphLaunch(child, cudaStreamGraphTailLaunch);
}

__global__ void mark_time(std::uint64_t* output, int index) {
  if (threadIdx.x == 0) {
    output[index] = globaltimer_ns();
  }
}

__global__ void delay_ns(int duration_ns) {
  if (threadIdx.x != 0 || duration_ns <= 0) {
    return;
  }
  const std::uint64_t start = globaltimer_ns();
  while (globaltimer_ns() - start < static_cast<std::uint64_t>(duration_ns)) {
    __nanosleep(128);
  }
}

cudaGraphExec_t capture_child(
    cudaStream_t stream, DispatchState* state, std::uint64_t* signals,
    std::uint64_t* child_start_ns, std::uint64_t* child_end_ns, int* results,
    int* seq_lens, const int* block_table, int* graph_slots, int* pat_deltas,
    int blocks_per_request, int request_count, int plan_slot, int work_ns,
    cudaGraph_t* graph_out) {
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  child_kernel<<<1, 128, 0, stream>>>(
      state, signals, child_start_ns, child_end_ns, results, seq_lens,
      block_table, graph_slots, pat_deltas, blocks_per_request, request_count,
      plan_slot, work_ns);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamEndCapture(stream, graph_out));
  cudaGraphExec_t graph_exec = nullptr;
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, *graph_out,
                                  cudaGraphInstantiateFlagDeviceLaunch));
  CUDA_CHECK(cudaGraphUpload(graph_exec, stream));
  return graph_exec;
}

double quantile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  const double position = fraction * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(position);
  const std::size_t upper = std::min(lower + 1, values.size() - 1);
  const double weight = position - static_cast<double>(lower);
  return values[lower] * (1.0 - weight) + values[upper] * weight;
}

double mean(const std::vector<double>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_args(argc, argv);
  const char* pmi_rank = std::getenv("PMI_RANK");
  if (pmi_rank == nullptr) {
    std::fprintf(stderr, "PMI_RANK is required\n");
    return 2;
  }
  const int selected_device = std::atoi(pmi_rank);
  CU_CHECK(cuInit(0));
  CUDA_CHECK(cudaSetDevice(selected_device));
  nvshmem_init();
  const int mype = nvshmem_my_pe();
  if (nvshmem_n_pes() != 2) {
    if (mype == 0) {
      std::fprintf(stderr, "this benchmark requires exactly two PEs\n");
    }
    nvshmem_finalize();
    return 2;
  }

  const int total = options.warmup + options.iterations;
  auto* descriptor =
      static_cast<StepDescriptor*>(nvshmem_malloc(sizeof(StepDescriptor)));
  auto* signals =
      static_cast<std::uint64_t*>(nvshmem_malloc(2 * sizeof(std::uint64_t)));
  if (descriptor == nullptr || signals == nullptr) {
    std::fprintf(stderr, "PE %d failed to allocate symmetric state\n", mype);
    nvshmem_global_exit(1);
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaMemsetAsync(descriptor, 0, sizeof(StepDescriptor), stream));
  CUDA_CHECK(cudaMemsetAsync(signals, 0, 2 * sizeof(std::uint64_t), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  nvshmem_barrier_all();

  if (mype == 0) {
    std::vector<StepDescriptor> host_descriptors(total + 1);
    for (int index = 0; index < total; ++index) {
      host_descriptors[index] = {
          static_cast<std::uint64_t>(index + 1),
          static_cast<std::uint32_t>(index & 1),
          0,
      };
    }
    host_descriptors[total] = {static_cast<std::uint64_t>(total + 1), 0, kStop};
    StepDescriptor* send_descriptors = nullptr;
    std::uint64_t* send_ns = nullptr;
    std::uint64_t* complete_ns = nullptr;
    CUDA_CHECK(cudaMalloc(&send_descriptors,
                          host_descriptors.size() * sizeof(StepDescriptor)));
    CUDA_CHECK(cudaMalloc(&send_ns, total * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMalloc(&complete_ns, total * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMemcpy(send_descriptors, host_descriptors.data(),
                          host_descriptors.size() * sizeof(StepDescriptor),
                          cudaMemcpyHostToDevice));

    for (int index = 0; index < total; ++index) {
      if (options.publish_gap_ns > 0) {
        delay_ns<<<1, 1, 0, stream>>>(options.publish_gap_ns);
        CUDA_CHECK(cudaGetLastError());
      }
      mark_time<<<1, 1, 0, stream>>>(send_ns, index);
      CUDA_CHECK(cudaGetLastError());
      nvshmemx_putmem_signal_on_stream(
          descriptor, send_descriptors + index, sizeof(StepDescriptor), signals,
          static_cast<std::uint64_t>(index + 1), NVSHMEM_SIGNAL_SET, 1, stream);
      nvshmemx_signal_wait_until_on_stream(
          signals + 1, NVSHMEM_CMP_GE, static_cast<std::uint64_t>(index + 1),
          stream);
      mark_time<<<1, 1, 0, stream>>>(complete_ns, index);
      CUDA_CHECK(cudaGetLastError());
    }
    nvshmemx_putmem_signal_on_stream(
        descriptor, send_descriptors + total, sizeof(StepDescriptor), signals,
        static_cast<std::uint64_t>(total + 1), NVSHMEM_SIGNAL_SET, 1, stream);
    nvshmemx_signal_wait_until_on_stream(signals + 1, NVSHMEM_CMP_GE,
                                         static_cast<std::uint64_t>(total + 1),
                                         stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<std::uint64_t> host_send_ns(total);
    std::vector<std::uint64_t> host_complete_ns(total);
    CUDA_CHECK(cudaMemcpy(host_send_ns.data(), send_ns,
                          total * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_complete_ns.data(), complete_ns,
                          total * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost));
    std::vector<double> roundtrip_us;
    roundtrip_us.reserve(options.iterations);
    for (int index = options.warmup; index < total; ++index) {
      roundtrip_us.push_back(
          static_cast<double>(host_complete_ns[index] - host_send_ns[index]) /
          1000.0);
    }
    std::printf(
        "{\"role\":\"projection\",\"status\":\"passed\","
        "\"iterations\":%d,\"publish_gap_ns\":%d,"
        "\"roundtrip_us_mean\":%.3f,"
        "\"roundtrip_us_p50\":%.3f,\"roundtrip_us_p99\":%.3f}\n",
        options.iterations, options.publish_gap_ns, mean(roundtrip_us),
        quantile(roundtrip_us, 0.50), quantile(roundtrip_us, 0.99));
    cudaFree(complete_ns);
    cudaFree(send_ns);
    cudaFree(send_descriptors);
  } else {
    DispatchState* state = nullptr;
    std::uint64_t* observed_ns = nullptr;
    std::uint64_t* child_start_ns = nullptr;
    std::uint64_t* child_end_ns = nullptr;
    int* results = nullptr;
    int* seq_lens = nullptr;
    int* block_table = nullptr;
    int* graph_slots = nullptr;
    int* pat_deltas = nullptr;
    constexpr int initial_seq_len = 15;
    const int blocks_per_request = (initial_seq_len + total + 15) / 16 + 1;
    CUDA_CHECK(cudaMalloc(&state, sizeof(DispatchState)));
    CUDA_CHECK(cudaMalloc(&observed_ns, total * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMalloc(&child_start_ns, total * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMalloc(&child_end_ns, total * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMalloc(&results, total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&seq_lens, options.requests * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&block_table,
                          options.requests * blocks_per_request * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&graph_slots, options.requests * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&pat_deltas, options.requests * sizeof(int)));
    const DispatchState initial_state{1, 1, 0, 0, 0};
    CUDA_CHECK(cudaMemcpy(state, &initial_state, sizeof(initial_state),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(observed_ns, 0, total * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMemset(child_start_ns, 0, total * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMemset(child_end_ns, 0, total * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMemset(results, 0xff, total * sizeof(int)));
    std::vector<int> host_seq_lens(options.requests, initial_seq_len);
    std::vector<int> host_block_table(options.requests * blocks_per_request);
    for (int row = 0; row < options.requests; ++row) {
      for (int block = 0; block < blocks_per_request; ++block) {
        host_block_table[row * blocks_per_request + block] =
            row * blocks_per_request + block;
      }
    }
    CUDA_CHECK(cudaMemcpy(seq_lens, host_seq_lens.data(),
                          options.requests * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(block_table, host_block_table.data(),
                          host_block_table.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(graph_slots, 0xff, options.requests * sizeof(int)));
    CUDA_CHECK(cudaMemset(pat_deltas, 0, options.requests * sizeof(int)));

    cudaGraph_t child_graphs[2]{};
    cudaGraphExec_t child_execs[2]{};
    for (int plan_slot = 0; plan_slot < 2; ++plan_slot) {
      child_execs[plan_slot] =
          capture_child(stream, state, signals, child_start_ns, child_end_ns,
                        results, seq_lens, block_table, graph_slots, pat_deltas,
                        blocks_per_request, options.requests, plan_slot,
                        options.child_work_ns, &child_graphs[plan_slot]);
    }
    cudaGraph_t dispatcher_graph = nullptr;
    cudaGraphExec_t dispatcher_exec = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    if (options.stream_wait) {
      dispatch_once_kernel<<<1, 32, 0, stream>>>(state, descriptor, signals,
                                                 observed_ns, child_execs[0],
                                                 child_execs[1]);
    } else {
      dispatcher_kernel<<<1, 32, 0, stream>>>(state, descriptor, signals,
                                              observed_ns, child_execs[0],
                                              child_execs[1]);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamEndCapture(stream, &dispatcher_graph));
    CUDA_CHECK(cudaGraphInstantiate(
        &dispatcher_exec, dispatcher_graph,
        options.stream_wait ? 0 : cudaGraphInstantiateFlagDeviceLaunch));
    CUDA_CHECK(cudaGraphUpload(dispatcher_exec, stream));
    if (options.stream_wait) {
      int stream_mem_ops_supported = 0;
      CU_CHECK(cuDeviceGetAttribute(
          &stream_mem_ops_supported,
          CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, selected_device));
      if (stream_mem_ops_supported == 0) {
        std::fprintf(stderr,
                     "GPU does not support 64-bit stream memory waits\n");
        nvshmem_global_exit(1);
      }
      int flush_remote_writes_supported = 0;
      CU_CHECK(cuDeviceGetAttribute(&flush_remote_writes_supported,
                                    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES,
                                    selected_device));
      const unsigned int wait_flags =
          CU_STREAM_WAIT_VALUE_GEQ |
          (flush_remote_writes_supported != 0 ? CU_STREAM_WAIT_VALUE_FLUSH : 0);
      for (int index = 0; index <= total; ++index) {
        CU_CHECK(cuStreamWaitValue64(reinterpret_cast<CUstream>(stream),
                                     reinterpret_cast<CUdeviceptr>(signals),
                                     static_cast<std::uint64_t>(index + 1),
                                     wait_flags));
        CUDA_CHECK(cudaGraphLaunch(dispatcher_exec, stream));
      }
    } else {
      CUDA_CHECK(cudaGraphLaunch(dispatcher_exec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DispatchState final_state{};
    std::vector<std::uint64_t> host_observed_ns(total);
    std::vector<std::uint64_t> host_child_start_ns(total);
    std::vector<std::uint64_t> host_child_end_ns(total);
    std::vector<int> host_results(total);
    std::vector<int> host_graph_slots(options.requests);
    std::vector<int> host_pat_deltas(options.requests);
    CUDA_CHECK(cudaMemcpy(&final_state, state, sizeof(final_state),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_observed_ns.data(), observed_ns,
                          total * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_child_start_ns.data(), child_start_ns,
                          total * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_child_end_ns.data(), child_end_ns,
                          total * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_results.data(), results, total * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_seq_lens.data(), seq_lens,
                          options.requests * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_graph_slots.data(), graph_slots,
                          options.requests * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_pat_deltas.data(), pat_deltas,
                          options.requests * sizeof(int),
                          cudaMemcpyDeviceToHost));
    std::vector<double> launch_us;
    std::vector<double> child_us;
    launch_us.reserve(options.iterations);
    child_us.reserve(options.iterations);
    std::uint32_t validation_errors = final_state.errors;
    for (int index = options.warmup; index < total; ++index) {
      launch_us.push_back(static_cast<double>(host_child_start_ns[index] -
                                              host_observed_ns[index]) /
                          1000.0);
      child_us.push_back(static_cast<double>(host_child_end_ns[index] -
                                             host_child_start_ns[index]) /
                         1000.0);
      if (host_results[index] != (index & 1)) {
        validation_errors += 1;
      }
    }
    for (int row = 0; row < options.requests; ++row) {
      const int final_prior_seq_len = initial_seq_len + total - 1;
      const int logical_block = final_prior_seq_len / 16;
      const int expected_slot =
          host_block_table[row * blocks_per_request + logical_block] * 16 +
          final_prior_seq_len % 16;
      if (host_seq_lens[row] != initial_seq_len + total ||
          host_graph_slots[row] != expected_slot ||
          host_pat_deltas[row] != total) {
        validation_errors += 1;
      }
    }
    std::printf(
        "{\"role\":\"attention\",\"status\":\"%s\","
        "\"wait_mode\":\"%s\","
        "\"iterations\":%d,\"device_graph_launch_us_mean\":%.3f,"
        "\"device_graph_launch_us_p50\":%.3f,"
        "\"device_graph_launch_us_p99\":%.3f,"
        "\"child_us_mean\":%.3f,\"requests\":%d,"
        "\"metadata_steps\":%d,\"errors\":%u}\n",
        validation_errors == 0 ? "passed" : "failed",
        options.stream_wait ? "stream" : "kernel", options.iterations,
        mean(launch_us), quantile(launch_us, 0.50), quantile(launch_us, 0.99),
        mean(child_us), options.requests, total, validation_errors);

    cudaGraphExecDestroy(dispatcher_exec);
    cudaGraphDestroy(dispatcher_graph);
    for (int plan_slot = 0; plan_slot < 2; ++plan_slot) {
      cudaGraphExecDestroy(child_execs[plan_slot]);
      cudaGraphDestroy(child_graphs[plan_slot]);
    }
    cudaFree(pat_deltas);
    cudaFree(graph_slots);
    cudaFree(block_table);
    cudaFree(seq_lens);
    cudaFree(results);
    cudaFree(child_end_ns);
    cudaFree(child_start_ns);
    cudaFree(observed_ns);
    cudaFree(state);
    if (validation_errors != 0) {
      nvshmem_global_exit(1);
    }
  }

  nvshmem_barrier_all();
  cudaStreamDestroy(stream);
  nvshmem_free(signals);
  nvshmem_free(descriptor);
  nvshmem_finalize();
  return 0;
}
