# Public reference review

No directly matching public defect found in the inspected sources. Do not infer a proven fix from version age or unrelated issues.

- AMD EPYC9005 networking tuning guide documents enabling10-bit Tag for additional outstanding requests. It does not establish10-bit Tag as faulty here: <https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/tuning-guides/58473_amd-epyc-9005-tg-windows-network.pdf>
- NVIDIA ConnectX-7 v28.47.1026 release notes examined: <https://docs.nvidia.com/networking/display/nvidia-connectx-7-adapter-cards-firmware-release-notes-v28-47-1026-october-2025-ga-release.1026%20(October%202025%20GA%20Release).pdf>
  The switch warm-reboot hang entry does not match this direct-root-port workload-time failure; the issue for MCX750500B-0D0K is a different part number. No claimed match. The listed later firmware/MKey issues do not establish causality for our endpoint AER0x14000.
- An AMD IDE tag-reuse security bulletin was returned by search; current evidence does not show PCIe IDE usage or this security issue, so excluded as a claimed explanation.
