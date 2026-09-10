# Latest 4P4D crash: reproduction and unresolved cause

Latest ordinary failure: `PAP-20260910-CROSS-HOST-PD-MATRIX/runs/20260910_crosshost_4p4d_c30_r2`.
Client started 2026-09-10 11:15:06.283 CST. Last persisted local records: approximately 11:15:41–42, about 36s after start. This is not the exact hardware failure timestamp; BMC event 174 subsequently supplied: Fatal asserted at 11:16:00 CST. Current boot began 11:43:01. Previous-boot kernel records contain no matched AER/Xid/fatal mechanism. Recovered remote logs show etcd connection failures after the local reboot, not a proven pre-failure RDMA cause.

Requested/effective mapping is local Decode GPUs4–7, remote Prefill GPUs0–3, 4P4D C30, 180 requests, Qwen3-8B. Startup audit passed. Frozen failed-run runner is byte-identical to the prior successful diagnostic runner. NIXL compatibility hash also matches. The original failure ran without the additional diagnostic supervisor, so recording overhead and execution history are differences; full inherited local process environment was not saved by that ordinary run and cannot now be reconstructed.

| Reproduction | Additional local periodic NVML calls | Requests | Client duration | Result |
|---|---|---|---|---|
| 20260910T034808Z_4p4d_r2_reproduce_r1 | on | 180/180 | 92.94s | passed |
| 20260910T035340Z_4p4d_no_nvml_poll_r1 | off | 180/180 | 94.98s | passed |

Both use original failed effective configuration and runner; startup mappings verified; residual failed-run processes identified by namespace and cleaned after preserving logs. Original AIPerf/DCGM and remote telemetry remain in both. Logs, sysfs AER, fsync and remote mirroring remain in both. Thus this tests only supervisor periodic local NVML polling, not all monitoring effects. No BIOS/register/power changes or error clearing performed. Zero Linux AER changes cannot rule out PCIe errors because root reporting was disabled and root-only status/BMC unavailable.

Measured conclusion: the latest workload did not reproduce the whole-host failure in these two short observations. Neither cross-NUMA by itself nor absence of supervisor NVML polling is established as a sufficient trigger. No hardware/driver root cause is established, and intermittent/cumulative faults remain possible.

Candidate requiring hardware validation: prior root lspci supplied by user showed ConnectX-7 and its root port MPS=512B, while GPU branches MPS=256B. This is NOT proof of an oversized transaction. Actual transaction size/error type and a controlled configuration experiment would be needed. Linux documents an all-device 128B MPS policy for P2P compatibility: <https://docs.kernel.org/admin-guide/kernel-parameters.html> (pci=pcie_bus_peer2peer). Host bridge P2P routing support remains platform-specific: <https://docs.kernel.org/driver-api/pci/p2pdma.html> . The parameter does not make unsupported routing work. No kernel command line or PCIe MPS was changed.

Next evidence needed: newest BMC Fatal timestamp/location/detail plus root snapshot captured by `sudo bash benchmarks/pap/experiments/e2e/PAP-20260910-PCIE-NUMA-DIAG/capture-root.sh`. That script only reads PCI/DMI/journal/BMC if ipmitool exists and writes logs; it never clears registers or changes configuration. Repeated risky runs without hardware error details may not resolve the mechanism.

## BMC events 174–181 supplied after reproduction

User supplied latest BMC events. With confirmed +8h correction: event174 PCIe Fatal Assert=11:16:00; event175 Critical=11:16:07. Relative to client start11:15:06.283, the BMC Fatal was reported 53.717s later. Last persisted application event11:15:42.023929 precedes the BMC Fatal by17.976071s. This gap does not establish where the request/GPU stopped or when the underlying fault began; buffering and BMC reporting latency are unmeasured.

Fatal Deassert11:38:33 was followed by S4/S5 at11:38:35 and power-up initiated restart at11:38:41. BIOS boot complete11:42:44 aligns with kernel start11:43:01 (17s). Do not interpret Deassert as spontaneous repair or infer power-loss causality from this table. No slot/BDF/location field supplied, so whether the newest failure hit Slot13/14 remains UNKNOWN. Need vendor event174 detail/raw SEL/location decode. Root snapshot directory was not present when checked in this turn.

## Root snapshot 20260910T040414Z inspected

User executed capture-root.sh. Boot135762b1-390d-4afe-8fbb-2de60d93937b, after latest crash reboot and two successful reproductions. All four GPU4–7 endpoints and sampled root ports have no set DevSta error flags/UESta/CESta; headers zero. NIC31:00.0 has DevSta CorrErr+, UnsupReq+ and CESta AdvNonFatalErr+, same flag pattern as earlier user-provided snapshot across a different boot, not an observed new transition. Cannot time or attribute these sticky bits. MPS remains NIC/root512B vs GPU/root256B. Linux root AER interrupt enables remain off. No BMC files generated (ipmitool not installed/available); latest event174 location still missing. Static post-reboot snapshot has not revealed the Fatal error mechanism. Avoid interpreting no recorded error as healthy hardware or assigning root cause to MPS difference without transaction evidence/controlled test.

## Latest BMC detailed location confirmed by user

Detailed-view event150 at displayed03:16:00 (corrected11:16:00 CST) identifies CPU1_P1_PCIE7 / Slot14 / 0000:e1:00.0, i.e. physical GPU6. It corresponds to summary-view event174; differing view numbers are not separate faults. Previous detailed event142 at displayed01:59:32 (09:59:32 CST) identifies the exact same branch. Thus the two latest confirmed fatal reports repeat on Slot14/GPU6; earlier Slot13/GPU7 report remains a separate historical observation. Repetition prioritizes GPU6 plus its slot/link/root-port/physical power path, not proof that GPU silicon alone or NIC alone is faulty. Successful short runs including GPU6 do not exclude an intermittent fault. Detailed restart event155 explicitly says Chassis Control command; not evidence of spontaneous PSU power loss.

Next discriminating experiment is branch/board localization with a fixed workload: equal-exposure repeated GPU sets with and without GPU6 can screen correlation; a powered-off documented swap of GPU6 with another GPU followed by PCI BDF-to-UUID verification can distinguish a fault following the board from one remaining at Slot14. Preserve CPU affinity, NIC and workload, recognizing placement changes may also alter physical route. No physical swap or new benchmark performed in response to this BMC update. AER error subtype still unavailable.

## Campaign reproduction confirmed at a different branch

Round3 of20260910T041346Z_fatal_repro produced BMC event158 at12:22:02 CST, CPU1_P0_PCIE8 / Slot13 / f1:00.0 / GPU7, unlike prior two Slot14/GPU6 reports. See runs/20260910T041346Z_fatal_repro_r03/postmortem/failure-report.md. Root cause remains unestablished.

## Same-NUMA repeated control also crashed

20260910T045130Z_node0_repeat round2 failed after round1 passed, local Decode0–3 verified. Last sample12:57:31.808, remote first timeout12:57:32.912 CST; user confirmed reboot. Cross-NUMA and active Decode6/7 are not necessary conditions. New BMC location/type pending; root cause not established. See the round2 postmortem report.

## BMC confirms same-NUMA PCIe Fatal on active GPU3

User detailed event166: displayed04:57:31, corrected12:57:31 CST, CPU0_P0_PCIE4 / Slot1 / 0000:71:00.0 / GPU3 (UUID GPU-e9dd15ab-ba98-f70a-f2a7-0cbdeae2f2d2). Thus this is a PCIe Fatal, not merely an application outage. It happened on an active CPU0 Decode branch with NIC also NUMA0. Previous reproduced event158 was CPU1_P0_PCIE8 / Slot13 / f1:00.0 / GPU7 during Decode4–7 load. Earlier Fatal events also implicated GPU6/e1. Reported branch changes with the active GPU group across sockets; this does not by itself identify the faulty component or rule out multiple physical faults. CPU1-only, GPU6/7-only and mandatory cross-NUMA explanations are insufficient. Same-NUMA NODE path still traverses different host bridges; not equivalent to same PCIe switch. Common GPU/NIC P2P behavior, NIC/driver/firmware/registration lifecycle and platform/power factors remain unresolved. Do not infer a last-worker-index bug from GPU3/7: prior GPU6 event is a counterexample.

BMC second-resolution report12:57:31 agrees with local final completed sample12:57:31.808 and first remote timeout sample12:57:32.912; this does not establish sub-second hardware ordering because snapshot fields are sequential and clocks/report latencies differ. BMC later indicates Chassis Control restart at13:27:07, BIOS boot complete13:31:20, kernel start13:31:38. Fault subtype still absent.

## Persistent hardware panic evidence recovered

Six ERST journal attachments consistently show Completion Timeout + Unexpected Completion (AER0x14000), firmware GHES fatal and kernel panic on GPU7/GPU6/GPU3. Earlier absence-of-subtype statements are superseded. Read hardware-audit/20260910T054218Z/findings.md and raw recovered logs. Physical/firmware cause remains unproven.

## Current BMC health collected

See hardware-audit/bmc-20260910T061142Z/findings.md. No asserted PSU3/main-power/fan/Sys_Health/PCIe_Status fault in current standard readings; red LED-specific cause remains unknown. CLI BMC clock matches CST, so prior8h correction is a web-display difference, not demonstrated RTC skew. mc info has protocol warnings despite exit0 and is qualified.

## Read-only runtime and memory-path comparison

See hardware-audit/runtime-comparison-20260910/findings.md. Loaded versions and key binaries match; no established plugin ABI failure. Actual service roles differ. UCX supports separate DMA-BUF GPU registration despite no nvidia_peermem, but historical MR/protocol selection was not traced. No further load run in this audit.
