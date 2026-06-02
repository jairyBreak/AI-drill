#!/bin/bash

echo "=============================================="
echo "FOUR SAMPLE CASES: Testing Congestion States"
echo "=============================================="

echo ""
echo "CASE 1: Normal (no congestion)"
echo "Loss=0%, Latency<20ms, qdepth_max all low"
tail -n +2 training_dataset_master.csv | awk -F',' '$24 == 0 && $22 < 20' | head -1 | \
awk -F',' '{
  printf "Ports 2-5 qdepth_max: %d, %d, %d, %d\n", $1, $4, $7, $10
  printf "Ports 2-5 mbps_mean: %.4f, %.4f, %.4f, %.4f\n", $2, $5, $8, $11
  printf "Ports 2-5 mbps_std: %.4f, %.4f, %.4f, %.4f\n", $3, $6, $9, $12
  printf "Total Load: %.2f Mbps\n", $13
  printf "Latency: %.3f ms | Jitter: %.3f ms | Loss: %.3f%%\n", $22, $23, $24
}'

echo ""
echo "CASE 2: Sustained Congestion on single port"
echo "Loss>10%, qdepth_max=64 on ports 2,3 but <10 on ports 4,5"
tail -n +2 training_dataset_master.csv | awk -F',' '$24 > 10 && $1 == 64 && $4 == 64 && $7 < 10 && $10 < 10' | head -1 | \
awk -F',' '{
  printf "Ports 2-5 qdepth_max: %d, %d, %d, %d\n", $1, $4, $7, $10
  printf "Ports 2-5 mbps_mean: %.4f, %.4f, %.4f, %.4f\n", $2, $5, $8, $11
  printf "Ports 2-5 mbps_std: %.4f, %.4f, %.4f, %.4f\n", $3, $6, $9, $12
  printf "Total Load: %.2f Mbps\n", $13
  printf "Latency: %.3f ms | Jitter: %.3f ms | Loss: %.3f%%\n", $22, $23, $24
}'

echo ""
echo "CASE 3: Burst Congestion (high latency/loss but lower qdepth)"
echo "Loss>5%, qdepth_max all <20, but high latency"
tail -n +2 training_dataset_master.csv | awk -F',' '$24 > 5 && $24 < 50 && $1 < 20 && $4 < 20 && $22 > 50' | head -1 | \
awk -F',' '{
  printf "Ports 2-5 qdepth_max: %d, %d, %d, %d\n", $1, $4, $7, $10
  printf "Ports 2-5 mbps_mean: %.4f, %.4f, %.4f, %.4f\n", $2, $5, $8, $11
  printf "Ports 2-5 mbps_std: %.4f, %.4f, %.4f, %.4f\n", $3, $6, $9, $12
  printf "Total Load: %.2f Mbps\n", $13
  printf "Latency: %.3f ms | Jitter: %.3f ms | Loss: %.3f%%\n", $22, $23, $24
}'

echo ""
echo "CASE 4: Non-congestion loss (link quality issues)"
echo "Loss>1% but qdepth_max all <5 and latency <50ms"
tail -n +2 training_dataset_master.csv | awk -F',' '$24 > 1 && $24 < 5 && $1 < 5 && $4 < 5 && $22 < 50' | head -1 | \
awk -F',' '{
  if(NF > 0) {
    printf "Ports 2-5 qdepth_max: %d, %d, %d, %d\n", $1, $4, $7, $10
    printf "Ports 2-5 mbps_mean: %.4f, %.4f, %.4f, %.4f\n", $2, $5, $8, $11
    printf "Ports 2-5 mbps_std: %.4f, %.4f, %.4f, %.4f\n", $3, $6, $9, $12
    printf "Total Load: %.2f Mbps\n", $13
    printf "Latency: %.3f ms | Jitter: %.3f ms | Loss: %.3f%%\n", $22, $23, $24
  } else {
    printf "(No samples match criteria)\n"
  }
}'

