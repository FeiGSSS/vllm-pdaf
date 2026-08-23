// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" int pap_nvshmem_graph_advance_epoch(std::uint64_t* epoch,
                                               cudaStream_t stream);
extern "C" int pap_nvshmem_graph_wait_signal(std::uint64_t* signal,
                                             const std::uint64_t* epoch,
                                             int layer_count, int layer_index,
                                             int generation_delta,
                                             cudaStream_t stream);
extern "C" int pap_nvshmem_graph_put_signal(
    void* destination, const void* source, std::size_t num_bytes,
    std::uint64_t* signal, const std::uint64_t* epoch, int layer_count,
    int layer_index, int peer, cudaStream_t stream);
#define BRIDGE_CHECK(call)                                            \
  do {                                                                \
    int status_ = (call);                                             \
    if (status_ != 0) {                                               \
      std::fprintf(stderr, "%s:%d bridge CUDA error: %d\n", __FILE__, \
                   __LINE__, status_);                                \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

#define CUDA_CHECK(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = (call);                                        \
    if (status_ != cudaSuccess) {                                        \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(status_));                         \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

struct Options {
  int layers = 36;
  int payload_bytes = 32 * 1024;
  int warmup = 10;
  int iterations = 100;
};

static int parse_positive(const char* name, const char* value) {
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0 || parsed > INT32_MAX) {
    std::fprintf(stderr, "%s must be a positive integer\n", name);
    std::exit(2);
  }
  return static_cast<int>(parsed);
}

static Options parse_args(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) {
      std::fprintf(stderr, "missing value for %s\n", argv[index]);
      std::exit(2);
    }
    if (std::strcmp(argv[index], "--layers") == 0) {
      options.layers = parse_positive("--layers", argv[index + 1]);
    } else if (std::strcmp(argv[index], "--payload-bytes") == 0) {
      options.payload_bytes =
          parse_positive("--payload-bytes", argv[index + 1]);
    } else if (std::strcmp(argv[index], "--warmup") == 0) {
      options.warmup = parse_positive("--warmup", argv[index + 1]);
    } else if (std::strcmp(argv[index], "--iterations") == 0) {
      options.iterations = parse_positive("--iterations", argv[index + 1]);
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", argv[index]);
      std::exit(2);
    }
  }
  if (options.payload_bytes % static_cast<int>(sizeof(int)) != 0) {
    std::fprintf(stderr, "--payload-bytes must be int aligned\n");
    std::exit(2);
  }
  return options;
}

__global__ void initialize_payload(int* payload, const std::uint64_t* epoch,
                                   int payload_elements, int layers,
                                   int layer) {
  const std::uint64_t generation =
      (*epoch - 1) * static_cast<std::uint64_t>(layers) +
      static_cast<std::uint64_t>(layer) + 1;
  for (int index = threadIdx.x; index < payload_elements; index += blockDim.x) {
    payload[index] = static_cast<int>(generation) + index;
  }
}

__global__ void increment_payload(int* payload, int payload_elements) {
  for (int index = threadIdx.x; index < payload_elements; index += blockDim.x) {
    payload[index] += 1;
  }
}

__global__ void validate_payload(const int* payload, const std::uint64_t* epoch,
                                 unsigned int* errors, int payload_elements,
                                 int layers, int layer) {
  const std::uint64_t generation =
      (*epoch - 1) * static_cast<std::uint64_t>(layers) +
      static_cast<std::uint64_t>(layer) + 1;
  for (int index = threadIdx.x; index < payload_elements; index += blockDim.x) {
    const int expected = static_cast<int>(generation) + index + 1;
    if (payload[index] != expected) {
      atomicAdd(errors, 1U);
    }
  }
}

int main(int argc, char** argv) {
  const Options options = parse_args(argc, argv);
  const char* pmi_rank = std::getenv("PMI_RANK");
  if (pmi_rank == nullptr) {
    std::fprintf(stderr, "PMI_RANK is required to select a GPU before init\n");
    return 2;
  }
  const int selected_device = std::atoi(pmi_rank);
  if (selected_device < 0) {
    std::fprintf(stderr, "PMI_RANK must be non-negative\n");
    return 2;
  }
  CUDA_CHECK(cudaSetDevice(selected_device));
  nvshmem_init();
  const int mype = nvshmem_my_pe();
  const int npes = nvshmem_n_pes();
  if (npes != 2) {
    if (mype == 0) {
      std::fprintf(stderr, "this benchmark requires exactly two PEs\n");
    }
    nvshmem_finalize();
    return 2;
  }

  const int local_pe = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  if (local_pe != selected_device) {
    std::fprintf(stderr, "PE %d selected GPU %d but NVSHMEM assigned GPU %d\n",
                 mype, selected_device, local_pe);
    nvshmem_finalize();
    return 2;
  }
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, local_pe));
  if (!properties.cooperativeLaunch) {
    std::fprintf(stderr, "PE %d GPU lacks cooperative launch\n", mype);
    nvshmem_finalize();
    return 2;
  }

  auto* payload = static_cast<int*>(nvshmem_malloc(options.payload_bytes));
  auto* signals =
      static_cast<std::uint64_t*>(nvshmem_malloc(2 * sizeof(std::uint64_t)));
  if (payload == nullptr || signals == nullptr) {
    std::fprintf(stderr, "PE %d failed to allocate symmetric memory\n", mype);
    nvshmem_global_exit(1);
  }
  std::uint64_t* epoch = nullptr;
  unsigned int* errors = nullptr;
  CUDA_CHECK(cudaMalloc(&epoch, sizeof(std::uint64_t)));
  CUDA_CHECK(cudaMalloc(&errors, sizeof(unsigned int)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaMemsetAsync(payload, 0, options.payload_bytes, stream));
  CUDA_CHECK(cudaMemsetAsync(signals, 0, 2 * sizeof(std::uint64_t), stream));
  CUDA_CHECK(cudaMemsetAsync(epoch, 0, sizeof(std::uint64_t), stream));
  CUDA_CHECK(cudaMemsetAsync(errors, 0, sizeof(unsigned int), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  nvshmem_barrier_all();

  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  BRIDGE_CHECK(pap_nvshmem_graph_advance_epoch(epoch, stream));
  const int payload_elements = options.payload_bytes / sizeof(int);
  constexpr int ready_qkv = 0;
  constexpr int ready_output = 1;
  const int peer = 1 - mype;
  for (int layer = 0; layer < options.layers; ++layer) {
    if (mype == 0) {
      initialize_payload<<<1, 256, 0, stream>>>(
          payload, epoch, payload_elements, options.layers, layer);
      CUDA_CHECK(cudaGetLastError());
      BRIDGE_CHECK(pap_nvshmem_graph_put_signal(
          payload, payload, options.payload_bytes, signals + ready_qkv, epoch,
          options.layers, layer, peer, stream));
      BRIDGE_CHECK(pap_nvshmem_graph_wait_signal(
          signals + ready_output, epoch, options.layers, layer, 0, stream));
      validate_payload<<<1, 256, 0, stream>>>(
          payload, epoch, errors, payload_elements, options.layers, layer);
      CUDA_CHECK(cudaGetLastError());
    } else {
      BRIDGE_CHECK(pap_nvshmem_graph_wait_signal(
          signals + ready_qkv, epoch, options.layers, layer, 0, stream));
      increment_payload<<<1, 256, 0, stream>>>(payload, payload_elements);
      CUDA_CHECK(cudaGetLastError());
      BRIDGE_CHECK(pap_nvshmem_graph_put_signal(
          payload, payload, options.payload_bytes, signals + ready_output,
          epoch, options.layers, layer, peer, stream));
    }
  }
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  std::size_t graph_nodes = 0;
  CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &graph_nodes));
  const int nodes_per_layer = mype == 0 ? 4 : 3;
  const std::size_t expected_nodes =
      static_cast<std::size_t>(options.layers * nodes_per_layer + 1);
  if (graph_nodes != expected_nodes) {
    std::fprintf(stderr, "PE %d captured %zu graph nodes, expected %d\n", mype,
                 graph_nodes, static_cast<int>(expected_nodes));
    nvshmem_global_exit(1);
  }
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  nvshmem_barrier_all();
  for (int iteration = 0; iteration < options.warmup; ++iteration) {
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  nvshmem_barrier_all();

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  const auto submit_start = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
  }
  const auto submit_end = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0F;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  unsigned int host_errors = 0;
  std::uint64_t host_epoch = 0;
  CUDA_CHECK(cudaMemcpy(&host_errors, errors, sizeof(host_errors),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&host_epoch, epoch, sizeof(host_epoch),
                        cudaMemcpyDeviceToHost));
  nvshmem_barrier_all();

  if (mype == 0) {
    const double submit_us =
        std::chrono::duration<double, std::micro>(submit_end - submit_start)
            .count() /
        options.iterations;
    const double step_us = elapsed_ms * 1000.0 / options.iterations;
    const double layer_us = step_us / options.layers;
    const double gib_per_second =
        (2.0 * options.payload_bytes * options.layers) / (step_us * 1.0e-6) /
        static_cast<double>(1ULL << 30);
    std::printf(
        "{\"status\":\"%s\",\"gpu\":\"%s\","
        "\"layers\":%d,\"graph_nodes\":%zu,\"payload_bytes\":%d,"
        "\"warmup\":%d,\"iterations\":%d,"
        "\"graph_launch_submit_us_per_step\":%.3f,"
        "\"gpu_step_us\":%.3f,\"gpu_layer_roundtrip_us\":%.3f,"
        "\"bidirectional_payload_gib_per_s\":%.3f,"
        "\"errors\":%u,\"final_epoch\":%llu}\n",
        host_errors == 0 ? "passed" : "failed", properties.name, options.layers,
        graph_nodes, options.payload_bytes, options.warmup, options.iterations,
        submit_us, step_us, layer_us, gib_per_second, host_errors,
        static_cast<unsigned long long>(host_epoch));
  }

  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaFree(errors);
  cudaFree(epoch);
  nvshmem_free(signals);
  nvshmem_free(payload);
  nvshmem_finalize();
  return host_errors == 0 ? 0 : 1;
}
