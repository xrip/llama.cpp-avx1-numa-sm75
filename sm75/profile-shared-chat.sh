#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
profiler=/home/xrip/tmp/ncu-old-compare-20260903/private/linux-desktop-glibc_2_11_3-x64/ncu
active=$(systemctl is-active cmp-idle-governor.service || true)
restore() {
    if [ "$active" = active ]; then
        sudo -n systemctl start cmp-idle-governor.service
    fi
}
trap restore EXIT
sudo -n systemctl stop cmp-idle-governor.service
sudo -n env GGML_TEST_SWIGLU_MMQ=1 GGML_CUDA_SM75_SWIGLU_Q8_1=1 "$profiler" \
    --section-folder /home/xrip/tmp/ncu-old-compare-20260903/root/usr/lib/x86_64-linux-gnu/nsight-compute/sections \
    --clock-control none --kernel-name regex:quantize_mmq_q8_1 --launch-count 1 \
    --section LaunchStats --section Occupancy --section SchedulerStats --section WarpStateStats \
    --section SpeedOfLight --section MemoryWorkloadAnalysis \
    -o "$root/fused-quant-metrics-p0" \
    "$root/build/bin/test-backend-ops" perf -b CUDA0 -o SWIGLU_MMQ \
    -p 'type_a=q6_K.*m=4096,n=2048,k=12288'
