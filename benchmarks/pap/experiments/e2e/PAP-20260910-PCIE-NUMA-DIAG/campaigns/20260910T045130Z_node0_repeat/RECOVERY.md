# Matched repeated NUMA0 control

Campaign 20260910T045130Z_node0_repeat; reference 20260910T041346Z_fatal_repro, which failed on round3 at Slot13/GPU7 after two passes.
Run up to10 rounds of same4P4D C30 180 requests; only workload config change is local Decode GPU0–3 instead of4–7. Remote Prefill0–3 unchanged. Runner, supervisor and witness hashes identical to reference; campaign wrapper only adds manifest validation and corrects dynamic recovery descriptions.
New boot differs from reference; no claim all initial hardware state identical. Do not call one pass stable. All original monitoring/fsync/rsync retained, with root-only capture optional and separately noted.
Each round's RECOVERY.md/current-round.txt and BASE/CURRENT_RUN locate evidence. Runtime mirror L20T:/home/fei/pd-crash-evidence/<run>/local-mirror/ plus independent witness.jsonl. Additional rejected postmortem upload from prior task is not retried here.
On failure stop workload and collect BMC event and previous boot journal before any new test. No auto reboot or power control. Preserve original logs and off-host copy separately. Never kill historical PIDs without verifying namespace/identity.
