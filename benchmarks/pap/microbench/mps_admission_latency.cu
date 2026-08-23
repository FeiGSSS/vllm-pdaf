// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#define CUDA_CHECK(call)                                                 \
  do {                                                                   \
    const cudaError_t status_ = (call);                                  \
    if (status_ != cudaSuccess) {                                        \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(status_));                         \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

struct Options {
  std::string role;
  std::string mode = "eager";
  std::string stream_priority = "high";
  std::string output;
  int expected_sms = 0;
  int duration_ms = 15000;
  int kernel_ms = 8;
  int queue_depth = 36;
  int iterations = 64;
  int warmup = 10;
  int interval_us = 1000;
  int graph_nodes = 4;
  int dynamic_smem_kib = 64;
  int cooperative = 0;
  int prepare_event = 0;
};

static int parse_positive(const char* name, const char* value) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0 || parsed > INT32_MAX) {
    std::fprintf(stderr, "%s must be a positive integer\n", name);
    std::exit(2);
  }
  return static_cast<int>(parsed);
}

static int parse_nonnegative(const char* name, const char* value) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 0 || parsed > INT32_MAX) {
    std::fprintf(stderr, "%s must be a non-negative integer\n", name);
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
    const char* name = argv[index];
    const char* value = argv[index + 1];
    if (std::strcmp(name, "--role") == 0) {
      options.role = value;
    } else if (std::strcmp(name, "--mode") == 0) {
      options.mode = value;
    } else if (std::strcmp(name, "--stream-priority") == 0) {
      options.stream_priority = value;
    } else if (std::strcmp(name, "--output") == 0) {
      options.output = value;
    } else if (std::strcmp(name, "--expected-sms") == 0) {
      options.expected_sms = parse_positive(name, value);
    } else if (std::strcmp(name, "--duration-ms") == 0) {
      options.duration_ms = parse_positive(name, value);
    } else if (std::strcmp(name, "--kernel-ms") == 0) {
      options.kernel_ms = parse_positive(name, value);
    } else if (std::strcmp(name, "--queue-depth") == 0) {
      options.queue_depth = parse_positive(name, value);
    } else if (std::strcmp(name, "--iterations") == 0) {
      options.iterations = parse_positive(name, value);
    } else if (std::strcmp(name, "--warmup") == 0) {
      options.warmup = parse_nonnegative(name, value);
    } else if (std::strcmp(name, "--interval-us") == 0) {
      options.interval_us = parse_nonnegative(name, value);
    } else if (std::strcmp(name, "--graph-nodes") == 0) {
      options.graph_nodes = parse_positive(name, value);
    } else if (std::strcmp(name, "--dynamic-smem-kib") == 0) {
      options.dynamic_smem_kib = parse_positive(name, value);
    } else if (std::strcmp(name, "--cooperative") == 0) {
      options.cooperative = parse_nonnegative(name, value);
    } else if (std::strcmp(name, "--prepare-event") == 0) {
      options.prepare_event = parse_nonnegative(name, value);
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", name);
      std::exit(2);
    }
  }
  if (options.role != "producer" && options.role != "marker") {
    std::fprintf(stderr, "--role must be producer or marker\n");
    std::exit(2);
  }
  if (options.mode != "eager" && options.mode != "graph") {
    std::fprintf(stderr, "--mode must be eager or graph\n");
    std::exit(2);
  }
  if (options.stream_priority != "normal" &&
      options.stream_priority != "high") {
    std::fprintf(stderr, "--stream-priority must be normal or high\n");
    std::exit(2);
  }
  if (options.cooperative != 0 && options.cooperative != 1) {
    std::fprintf(stderr, "--cooperative must be 0 or 1\n");
    std::exit(2);
  }
  if (options.prepare_event != 0 && options.prepare_event != 1) {
    std::fprintf(stderr, "--prepare-event must be 0 or 1\n");
    std::exit(2);
  }
  return options;
}

__global__ void burn_kernel(float* output, std::uint64_t target_cycles,
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

__global__ void marker_kernel(unsigned long long* sink) {
  if (threadIdx.x == 0) {
    atomicAdd(sink, 1ULL);
  }
}

__global__ void prepare_kernel(unsigned char* data, std::size_t size) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < size) {
    data[index] ^= 1;
  }
}

static void launch_marker_kernel(unsigned long long* sink, cudaStream_t stream,
                                 bool cooperative) {
  if (!cooperative) {
    marker_kernel<<<1, 32, 0, stream>>>(sink);
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  void* arguments[] = {&sink};
  CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(marker_kernel),
                                         dim3(1), dim3(32), arguments, 0,
                                         stream));
}

static cudaStream_t create_stream(const Options& options) {
  int least_priority = 0;
  int greatest_priority = 0;
  CUDA_CHECK(
      cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
  const int priority =
      options.stream_priority == "high" ? greatest_priority : 0;
  cudaStream_t stream;
  CUDA_CHECK(
      cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority));
  return stream;
}

static void audit_device(const Options& options, cudaDeviceProp* properties) {
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaGetDeviceProperties(properties, 0));
  if (options.expected_sms > 0 &&
      properties->multiProcessorCount != options.expected_sms) {
    std::fprintf(stderr, "static MPS exposed %d SMs; expected %d\n",
                 properties->multiProcessorCount, options.expected_sms);
    std::exit(2);
  }
}

static int run_producer(const Options& options) {
  cudaDeviceProp properties{};
  audit_device(options, &properties);
  const int blocks = properties.multiProcessorCount;
  constexpr int threads = 256;
  const int smem_bytes = options.dynamic_smem_kib * 1024;
  if (smem_bytes > properties.sharedMemPerBlockOptin) {
    std::fprintf(stderr,
                 "requested %d dynamic shared-memory bytes; device max %zu\n",
                 smem_bytes, properties.sharedMemPerBlockOptin);
    return 2;
  }
  CUDA_CHECK(cudaFuncSetAttribute(
      burn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

  float* output = nullptr;
  CUDA_CHECK(cudaMalloc(&output, blocks * threads * sizeof(float)));
  cudaStream_t stream = create_stream(options);
  const std::uint64_t target_cycles =
      static_cast<std::uint64_t>(properties.clockRate) * options.kernel_ms;

  burn_kernel<<<blocks, threads, smem_bytes, stream>>>(output, target_cycles,
                                                       threads);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const auto start = std::chrono::steady_clock::now();
  int batches = 0;
  int launches = 0;
  while (std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
             .count() < options.duration_ms) {
    for (int index = 0; index < options.queue_depth; ++index) {
      burn_kernel<<<blocks, threads, smem_bytes, stream>>>(
          output, target_cycles, threads);
      CUDA_CHECK(cudaGetLastError());
      ++launches;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ++batches;
  }
  const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start)
                                .count();

  std::printf(
      "{\"role\":\"producer\",\"status\":\"passed\","
      "\"gpu\":\"%s\",\"visible_sms\":%d,\"kernel_ms\":%d,"
      "\"queue_depth\":%d,\"launches\":%d,\"batches\":%d,"
      "\"elapsed_ms\":%.3f}\n",
      properties.name, properties.multiProcessorCount, options.kernel_ms,
      options.queue_depth, launches, batches, elapsed_ms);
  std::fflush(stdout);

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(output));
  return 0;
}

static double percentile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  const std::size_t index =
      static_cast<std::size_t>(std::ceil(fraction * values.size())) - 1;
  return values[std::min(index, values.size() - 1)];
}

static int run_marker(const Options& options) {
  cudaDeviceProp properties{};
  audit_device(options, &properties);
  unsigned long long* sink = nullptr;
  CUDA_CHECK(cudaMalloc(&sink, sizeof(*sink)));
  CUDA_CHECK(cudaMemset(sink, 0, sizeof(*sink)));
  cudaStream_t stream = create_stream(options);
  cudaStream_t prepare_stream = nullptr;
  cudaEvent_t prepare_done = nullptr;
  unsigned char* prepare_host = nullptr;
  unsigned char* prepare_device = nullptr;
  constexpr std::size_t prepare_bytes = 20916;
  if (options.prepare_event != 0) {
    CUDA_CHECK(
        cudaStreamCreateWithFlags(&prepare_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&prepare_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaMallocHost(&prepare_host, prepare_bytes));
    CUDA_CHECK(cudaMalloc(&prepare_device, prepare_bytes));
    std::memset(prepare_host, 1, prepare_bytes);
  }

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  if (options.mode == "graph") {
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int node = 0; node < options.graph_nodes; ++node) {
      launch_marker_kernel(sink, stream, options.cooperative != 0);
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  }

  const auto launch_once = [&]() {
    if (options.prepare_event != 0) {
      CUDA_CHECK(cudaMemcpyAsync(prepare_device, prepare_host, prepare_bytes,
                                 cudaMemcpyHostToDevice, prepare_stream));
      prepare_kernel<<<21, 128, 0, prepare_stream>>>(prepare_device,
                                                     prepare_bytes);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventRecord(prepare_done, prepare_stream));
      CUDA_CHECK(cudaStreamWaitEvent(stream, prepare_done, 0));
    }
    if (options.mode == "graph") {
      CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    } else {
      for (int node = 0; node < options.graph_nodes; ++node) {
        launch_marker_kernel(sink, stream, options.cooperative != 0);
      }
    }
    CUDA_CHECK(cudaGetLastError());
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
  const char* client_priority = std::getenv("CUDA_MPS_CLIENT_PRIORITY");
  std::printf(
      "{\"role\":\"marker\",\"status\":\"passed\","
      "\"gpu\":\"%s\",\"visible_sms\":%d,\"mode\":\"%s\","
      "\"graph_nodes\":%d,\"cooperative\":%d,\"prepare_event\":%d,"
      "\"iterations\":%d,"
      "\"connections\":\"%s\",\"client_priority\":\"%s\","
      "\"mean_us\":%.3f,\"p50_us\":%.3f,\"p95_us\":%.3f,"
      "\"p99_us\":%.3f,\"max_us\":%.3f}\n",
      properties.name, properties.multiProcessorCount, options.mode.c_str(),
      options.graph_nodes, options.cooperative, options.prepare_event,
      options.iterations, connections == nullptr ? "unset" : connections,
      client_priority == nullptr ? "unset" : client_priority,
      sum / latency_us.size(), percentile(latency_us, 0.50),
      percentile(latency_us, 0.95), percentile(latency_us, 0.99),
      *std::max_element(latency_us.begin(), latency_us.end()));
  std::fflush(stdout);

  if (graph_exec != nullptr) {
    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  }
  if (graph != nullptr) {
    CUDA_CHECK(cudaGraphDestroy(graph));
  }
  if (prepare_done != nullptr) {
    CUDA_CHECK(cudaEventDestroy(prepare_done));
  }
  if (prepare_stream != nullptr) {
    CUDA_CHECK(cudaStreamDestroy(prepare_stream));
  }
  if (prepare_device != nullptr) {
    CUDA_CHECK(cudaFree(prepare_device));
  }
  if (prepare_host != nullptr) {
    CUDA_CHECK(cudaFreeHost(prepare_host));
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(sink));
  return 0;
}

int main(int argc, char** argv) {
  const Options options = parse_args(argc, argv);
  if (options.role == "producer") {
    return run_producer(options);
  }
  return run_marker(options);
}
