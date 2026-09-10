# Repeated whole-host fatal reproduction

User explicitly requested trying to crash machine by reproducing the failing workload with detailed monitoring.
Ten sequential rounds of original 4P4D C30 180-request workload. Stops on any unsuccessful round; never reboots automatically.
Each round has separate frozen config, logs, hardware samples, SHA256 and L20T:/home/fei/pd-crash-evidence/<run>/ mirror plus independent witness.
CURRENT_RUN and current-round.txt locate latest round. Read its RECOVERY.md before cleanup. Missing final status is interrupted/unknown, never passed.
Root watcher reads only; availability depends on user sudo launch. Kernel/AER observation alone cannot establish cause; after crash collect BMC error location/type and prior-boot journal before next load.
Fixed GPU4–7 identity, same active NIC ibp49s0:1, remote mlx5_0:1. Latest two known Fatal locations Slot14/e1/GPU6; no root cause established.
