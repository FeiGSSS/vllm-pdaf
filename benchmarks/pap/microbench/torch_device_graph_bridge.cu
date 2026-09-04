// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

__global__ void launch_child(cudaGraphExec_t child) {
  if (threadIdx.x == 0) {
    cudaGraphLaunch(child, cudaStreamGraphTailLaunch);
  }
}

}  // namespace

extern "C" int pap_torch_device_graph_smoke(std::uintptr_t raw_graph_handle,
                                            std::uintptr_t stream_handle,
                                            int iterations, float* elapsed_ms) {
  if (raw_graph_handle == 0 || stream_handle == 0 || iterations <= 0 ||
      elapsed_ms == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  auto raw_graph = reinterpret_cast<cudaGraph_t>(raw_graph_handle);
  auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
  cudaGraph_t child_graph = nullptr;
  cudaGraphExec_t child_exec = nullptr;
  cudaGraph_t launcher_graph = nullptr;
  cudaGraphExec_t launcher_exec = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t end = nullptr;
  const char* failure_stage = "cudaGraphClone";

  cudaError_t status = cudaGraphClone(&child_graph, raw_graph);
  if (status == cudaSuccess) {
    std::size_t node_count = 0;
    status = cudaGraphGetNodes(child_graph, nullptr, &node_count);
    std::vector<cudaGraphNode_t> nodes(node_count);
    if (status == cudaSuccess && node_count > 0) {
      status = cudaGraphGetNodes(child_graph, nodes.data(), &node_count);
    }
    for (std::size_t index = 0; status == cudaSuccess && index < node_count;
         ++index) {
      cudaGraphNodeType type{};
      status = cudaGraphNodeGetType(nodes[index], &type);
      if (status == cudaSuccess) {
        std::fprintf(stderr, "torch graph node %zu type %d\n", index,
                     static_cast<int>(type));
      }
    }
  }
  failure_stage = "cudaGraphInstantiate(device)";
  if (status == cudaSuccess) {
    status = cudaGraphInstantiate(&child_exec, child_graph,
                                  cudaGraphInstantiateFlagDeviceLaunch);
  }
  failure_stage = "cudaGraphUpload(child)";
  if (status == cudaSuccess) {
    status = cudaGraphUpload(child_exec, stream);
  }
  failure_stage = "cudaStreamBeginCapture(launcher)";
  if (status == cudaSuccess) {
    status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  }
  failure_stage = "launch_child";
  if (status == cudaSuccess) {
    launch_child<<<1, 32, 0, stream>>>(child_exec);
    status = cudaGetLastError();
  }
  failure_stage = "cudaStreamEndCapture(launcher)";
  if (status == cudaSuccess) {
    status = cudaStreamEndCapture(stream, &launcher_graph);
  }
  failure_stage = "cudaGraphInstantiate(launcher)";
  if (status == cudaSuccess) {
    status = cudaGraphInstantiate(&launcher_exec, launcher_graph, 0);
  }
  failure_stage = "events or launch";
  if (status == cudaSuccess) {
    status = cudaEventCreate(&start);
  }
  if (status == cudaSuccess) {
    status = cudaEventCreate(&end);
  }
  if (status == cudaSuccess) {
    status = cudaEventRecord(start, stream);
  }
  for (int index = 0; status == cudaSuccess && index < iterations; ++index) {
    status = cudaGraphLaunch(launcher_exec, stream);
  }
  if (status == cudaSuccess) {
    status = cudaEventRecord(end, stream);
  }
  if (status == cudaSuccess) {
    status = cudaEventSynchronize(end);
  }
  if (status == cudaSuccess) {
    status = cudaEventElapsedTime(elapsed_ms, start, end);
  }
  if (status != cudaSuccess) {
    std::fprintf(stderr, "torch device graph failed at %s: %s\n", failure_stage,
                 cudaGetErrorString(status));
  }

  if (end != nullptr) {
    cudaEventDestroy(end);
  }
  if (start != nullptr) {
    cudaEventDestroy(start);
  }
  if (launcher_exec != nullptr) {
    cudaGraphExecDestroy(launcher_exec);
  }
  if (launcher_graph != nullptr) {
    cudaGraphDestroy(launcher_graph);
  }
  if (child_exec != nullptr) {
    cudaGraphExecDestroy(child_exec);
  }
  if (child_graph != nullptr) {
    cudaGraphDestroy(child_graph);
  }
  return static_cast<int>(status);
}
