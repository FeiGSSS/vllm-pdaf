// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

extern "C" int pap_nvshmem_graph_advance_epoch(std::uint64_t* epoch,
                                               cudaStream_t stream);
extern "C" int pap_nvshmem_device_bridge_init(unsigned int flags,
                                              nvshmemx_init_attr_t* attributes);
extern "C" void pap_nvshmem_device_bridge_finalize();
extern "C" int pap_nvshmem_graph_wait_signal(std::uint64_t* signal,
                                             const std::uint64_t* epoch,
                                             int layer_count, int layer_index,
                                             int generation_delta,
                                             cudaStream_t stream);
extern "C" int pap_nvshmem_graph_put_signal(
    void* destination, const void* source, std::size_t num_bytes,
    std::uint64_t* signal, const std::uint64_t* epoch, int layer_count,
    int layer_index, int peer, cudaStream_t stream);

#define CUDA_CHECK(call)                                                 \
  do {                                                                   \
    const cudaError_t status_ = (call);                                  \
    if (status_ != cudaSuccess) {                                        \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(status_));                         \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

#define BRIDGE_CHECK(call)                                                 \
  do {                                                                     \
    const int status_ = (call);                                            \
    if (status_ != 0) {                                                    \
      std::fprintf(stderr, "%s:%d bridge error: %d\n", __FILE__, __LINE__, \
                   status_);                                               \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

struct Options {
  std::string output;
  int expected_sms = 12;
  int iterations = 256;
  int warmup = 10;
  int interval_us = 10000;
  int layers = 36;
  int compute_us = 0;
};

static int parse_nonnegative(const char* name, const char* value) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 0 || parsed > INT32_MAX) {
    std::fprintf(stderr, "%s must be a non-negative integer\n", name);
    std::exit(2);
  }
  return static_cast<int>(parsed);
}

static int parse_positive(const char* name, const char* value) {
  const int value_int = parse_nonnegative(name, value);
  if (value_int == 0) {
    std::fprintf(stderr, "%s must be positive\n", name);
    std::exit(2);
  }
  return value_int;
}

static Options parse_args(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) {
      std::fprintf(stderr, "missing value for %s\n", argv[index]);
      std::exit(2);
    }
    const char* name = argv[index];
    const char* value = argv[index + 1];
    if (std::strcmp(name, "--output") == 0) {
      options.output = value;
    } else if (std::strcmp(name, "--expected-sms") == 0) {
      options.expected_sms = parse_positive(name, value);
    } else if (std::strcmp(name, "--iterations") == 0) {
      options.iterations = parse_positive(name, value);
    } else if (std::strcmp(name, "--warmup") == 0) {
      options.warmup = parse_nonnegative(name, value);
    } else if (std::strcmp(name, "--interval-us") == 0) {
      options.interval_us = parse_nonnegative(name, value);
    } else if (std::strcmp(name, "--layers") == 0) {
      options.layers = parse_positive(name, value);
    } else if (std::strcmp(name, "--compute-us") == 0) {
      options.compute_us = parse_nonnegative(name, value);
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", name);
      std::exit(2);
    }
  }
  return options;
}

__global__ void marker_compute(int* payload) {
  if (threadIdx.x == 0) {
    *payload += 1;
  }
}

__global__ void attention_burn_kernel(float* output,
                                      std::uint64_t target_cycles,
                                      int output_stride) {
  extern __shared__ unsigned char shared[];
  if (threadIdx.x == 0) {
    shared[0] = static_cast<unsigned char>(blockIdx.x);
  }
  __syncthreads();
  float value = static_cast<float>(blockIdx.x + threadIdx.x + 1);
  const std::uint64_t start = clock64();
  while (clock64() - start < target_cycles) {
    value = fmaf(value, 1.000000119F, 0.000000119F);
  }
  output[blockIdx.x * output_stride + threadIdx.x] =
      value + static_cast<float>(shared[0]);
}

static double percentile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  const std::size_t index =
      static_cast<std::size_t>(std::ceil(fraction * values.size())) - 1;
  return values[std::min(index, values.size() - 1)];
}

int main(int argc, char** argv) {
  const Options options = parse_args(argc, argv);
  nvshmem_init();
  if (pap_nvshmem_device_bridge_init(0, nullptr) != 0) {
    std::fprintf(stderr, "failed to initialize NVSHMEM bridge\n");
    nvshmem_finalize();
    return 2;
  }
  const int mype = nvshmem_my_pe();
  if (nvshmem_n_pes() != 2) {
    if (mype == 0) {
      std::fprintf(stderr, "NVSHMEM admission marker requires two PEs\n");
    }
    pap_nvshmem_device_bridge_finalize();
    nvshmem_finalize();
    return 2;
  }

  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  if (properties.multiProcessorCount != options.expected_sms) {
    std::fprintf(stderr, "static MPS exposed %d SMs; expected %d\n",
                 properties.multiProcessorCount, options.expected_sms);
    pap_nvshmem_device_bridge_finalize();
    nvshmem_finalize();
    return 2;
  }

  auto* payload = static_cast<int*>(nvshmem_malloc(2 * sizeof(int)));
  auto* signals =
      static_cast<std::uint64_t*>(nvshmem_malloc(2 * sizeof(std::uint64_t)));
  std::uint64_t* epoch = nullptr;
  float* burn_output = nullptr;
  CUDA_CHECK(cudaMalloc(&epoch, sizeof(*epoch)));
  constexpr int burn_threads = 256;
  CUDA_CHECK(cudaMalloc(&burn_output, properties.multiProcessorCount *
                                          burn_threads * sizeof(float)));
  constexpr int burn_smem_bytes = 64 * 1024;
  CUDA_CHECK(cudaFuncSetAttribute(attention_burn_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  burn_smem_bytes));
  const std::uint64_t burn_cycles =
      static_cast<std::uint64_t>(properties.clockRate) * options.compute_us /
      1000;
  cudaStream_t stream;
  int least_priority = 0;
  int greatest_priority = 0;
  CUDA_CHECK(
      cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
  CUDA_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking,
                                          greatest_priority));
  CUDA_CHECK(cudaMemsetAsync(payload, 0, 2 * sizeof(*payload), stream));
  CUDA_CHECK(cudaMemsetAsync(signals, 0xff, 2 * sizeof(*signals), stream));
  CUDA_CHECK(cudaMemsetAsync(epoch, 0, sizeof(*epoch), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  nvshmem_barrier_all();

  if (mype != 0) {
    nvshmem_barrier_all();
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(burn_output));
    CUDA_CHECK(cudaFree(epoch));
    nvshmem_free(signals);
    nvshmem_free(payload);
    pap_nvshmem_device_bridge_finalize();
    nvshmem_finalize();
    return 0;
  }

  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  BRIDGE_CHECK(pap_nvshmem_graph_advance_epoch(epoch, stream));
  for (int layer = 0; layer < options.layers; ++layer) {
    BRIDGE_CHECK(pap_nvshmem_graph_wait_signal(signals, epoch, options.layers,
                                               layer, 0, stream));
    if (options.compute_us > 0) {
      attention_burn_kernel<<<properties.multiProcessorCount, burn_threads,
                              burn_smem_bytes, stream>>>(
          burn_output, burn_cycles, burn_threads);
    } else {
      marker_compute<<<1, 32, 0, stream>>>(payload);
    }
    CUDA_CHECK(cudaGetLastError());
    BRIDGE_CHECK(pap_nvshmem_graph_put_signal(
        payload + 1, payload, sizeof(*payload), signals + 1, epoch,
        options.layers, layer, 1, stream));
  }
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  const auto launch_once = [&]() {
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  };
  for (int iteration = 0; iteration < options.warmup; ++iteration) {
    launch_once();
  }

  std::vector<double> latency_us;
  latency_us.reserve(options.iterations);
  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    launch_once();
    const auto stop = std::chrono::steady_clock::now();
    latency_us.push_back(
        std::chrono::duration<double, std::micro>(stop - start).count());
    if (options.interval_us > 0) {
      std::this_thread::sleep_for(
          std::chrono::microseconds(options.interval_us));
    }
  }

  if (!options.output.empty()) {
    std::ofstream output(options.output);
    output << "iteration,submit_to_complete_us\n";
    for (std::size_t index = 0; index < latency_us.size(); ++index) {
      output << index << ',' << latency_us[index] << '\n';
    }
  }
  const double sum = std::accumulate(latency_us.begin(), latency_us.end(), 0.0);
  const char* connections = std::getenv("CUDA_DEVICE_MAX_CONNECTIONS");
  std::printf(
      "{\"role\":\"nvshmem_marker\",\"status\":\"passed\","
      "\"gpu\":\"%s\",\"visible_sms\":%d,\"layers\":%d,"
      "\"graph_nodes\":%d,\"compute_us\":%d,\"iterations\":%d,"
      "\"connections\":\"%s\","
      "\"mean_us\":%.3f,\"p50_us\":%.3f,\"p95_us\":%.3f,"
      "\"p99_us\":%.3f,\"max_us\":%.3f}\n",
      properties.name, properties.multiProcessorCount, options.layers,
      1 + 3 * options.layers, options.compute_us, options.iterations,
      connections == nullptr ? "unset" : connections, sum / latency_us.size(),
      percentile(latency_us, 0.50), percentile(latency_us, 0.95),
      percentile(latency_us, 0.99),
      *std::max_element(latency_us.begin(), latency_us.end()));
  std::fflush(stdout);
  nvshmem_barrier_all();

  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(burn_output));
  CUDA_CHECK(cudaFree(epoch));
  nvshmem_free(signals);
  nvshmem_free(payload);
  pap_nvshmem_device_bridge_finalize();
  nvshmem_finalize();
  return 0;
}
