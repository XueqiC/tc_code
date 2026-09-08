#!/usr/bin/env bash
# Wait for both B13843 lanes to finish, then evaluate the B5537 adapters on the same Blackwell GPUs.
P=/home/xueqi/hq/projects/tc-alignment-table1; cd "$P"
until grep -q 'LANE COMPLETE' logs/table1_lane_gpu4.log 2>/dev/null && grep -q 'LANE COMPLETE' logs/table1_lane_gpu1.log 2>/dev/null; do sleep 600; done
echo "[after] $(date -u +%FT%TZ) B13843 lanes complete; starting B5537 evaluation lanes"
nohup bash tools/table1_bfcl_lane.sh 4 GPU-aaebd5af-4015-b475-52dd-5185ca7bdb52 5537 sft:0 sft:1 sft:2 sad:0 sad:1 sad:2 ddpo:0 ddpo:1 ddpo:2 > logs/table1_lane_gpu4_B5537.log 2>&1 &
nohup bash tools/table1_bfcl_lane.sh 1 GPU-97762062-28db-d85e-cb46-294b082f94df 5537 bbopd:0 bbopd:1 bbopd:2 pbsd_insp:0 pbsd_insp:1 pbsd_insp:2 pbsd_agent:0 pbsd_agent:1 pbsd_agent:2 > logs/table1_lane_gpu1_B5537.log 2>&1 &
echo "[after] B5537 lanes launched"
