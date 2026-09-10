# PCIe fatal completion errors: request for platform/GPU/NIC vendor analysis

Problem: Repeated cross-host vLLM/Dynamo PD workloads cause system-level PCIe fatal errors and Linux GHES kernel panic on NF5468-L20B. Reproduced with local Decode GPUs4–7 (NIC cross-NUMA) and GPUs0–3 (NIC same-NUMA). Eight independent TP1 workers,4 local Decode and4 remote Prefill, Qwen3-8B, NIXL/UCX RDMA. Some rounds complete and later rounds fail. No established single-card/slot or cross-NUMA-only root cause.

Hardware/software inventory: IEIT SYSTEMS NF5468-A8-A0-R0-00, two AMD EPYC9575F; BIOS01.07.00 dated2025-10-16, microcode0xb002161;8 NVIDIA L20(10de:26ba), VBIOS95.02.7C.00.06; NVIDIA open kernel driver595.84; Ubuntu kernel7.0.0-31-generic; Mellanox ConnectX-7 MCX75310AAS-NEAT / PSID MT_0000000838 / firmware28.43.2026 / PCI31:00.0, active400Gbps NDR IB; vLLM0.26.0, Dynamo1.4.1, UCX1.22.0 custom RDMA build. NIC NUMA0. GPUs0–3 NUMA0 NODE to NIC, GPUs4–7 NUMA1 SYS to NIC; endpoints directly under separate AMD root ports, no shared PCIe switch in those paths.

Decisive hardware evidence: six recovered ERST/pstore panic archives, endpoint sequence f1,f1,e1,e1,f1,71, all APEI GHES error source515, fatal PCIe error, uncorrectable status0x00014000, uncorrectable mask0, severity0x00476030, correctable status0x00008000, TLP header zeros. Decode: Completion Timeout + Unexpected Completion, both fatal/unmasked. Header Log Overflow also set. Exact log fragments and full recovered dmesg attached.

CPU NUMA locality/working GPU changes do not eliminate failure. GPU aggregate corrected/uncorrected ECC and current EDAC counts0; current rebooted mlx5 devlink health reporters healthy/error0. These snapshots do not exclude the observed transaction errors. Root AER interrupt enables/DPC trigger off under firmware ownership. amd_iommu=off in failed boots. ASPM disabled. Endpoint/root10BitTagReq enabled and capability advertised; this setting alone is not evidence of a bug. NIC path MPS512, GPU paths256; no Malformed TLP bit observed, so MPS mismatch is an unproven candidate. No BIOS/PCI register configuration was changed during diagnostics.

Requested vendor actions:

1. IEIT/AMD: confirm supported BIOS/AGESA/microcode baseline for this board revision + EPYC9575F + L20 + ConnectX-7. Provide any documented NBIO/PCIe completion routing or10-bit Tag errata matching0x14000 under peer DMA.
2. NVIDIA GPU: determine whether the reported L20 AER combination indicates a known GPU PCIe requester/tag tracking issue with this VBIOS/driver/platform. Identify what additional non-destructive dump captures requester/completer ID, Tag, address and original CPER/TLP data; current header is all zeros.
3. NVIDIA Networking: assess ConnectX-7 PSID MT_0000000838 firmware28.43.2026 interaction with this peer-memory path; specify an applicable issue ID and OEM-approved version if one exists. An unrelated later release is not sufficient evidence of a fix.
4. Provide a targeted next diagnostic or firmware fix with expected evidence and rollback requirements. Do not disable Fatal reporting or panic as a substitute for fixing PCIe errors.

Current conclusion: immediate machine-stop mechanism is established; the underlying component/firmware cause is NOT. This package is prepared locally and has NOT been sent to any vendor.
