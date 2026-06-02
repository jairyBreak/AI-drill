#!/bin/bash

echo "=============================================="
echo "Queue Depth Analysis (qdepth_max values)"
echo "=============================================="

# Extract qdepth columns and analyze
tail -n +2 training_dataset_master.csv | cut -d',' -f1,4,7,10 | \
awk -F',' '{
  for(i=1;i<=NF;i++) {
    val=$i
    if(val != "" && val != "0") {
      if(!(i in min) || val < min[i]) min[i]=val
      if(!(i in max) || val > max[i]) max[i]=val
      sum[i]+=val; count[i]++
    }
  }
}
END {
  ports=("port2" "port3" "port4" "port5")
  for(i=1;i<=4;i++) {
    printf "Port %d qdepth_max: min=%d, max=%d, count=%d\n", i+1, min[i], max[i], count[i]
  }
}'

echo ""
echo "=============================================="
echo "Loss Rate Distribution"
echo "=============================================="

tail -n +2 training_dataset_master.csv | cut -d',' -f24 | \
awk '{
  if($1 == 0) zero++
  else if($1 > 0 && $1 <= 1) low++
  else if($1 > 1 && $1 <= 10) med++
  else high++
}
END {
  total=zero+low+med+high
  printf "Zero loss (0%%): %d (%.1f%%)\n", zero, 100*zero/total
  printf "Low loss (0-1%%): %d (%.1f%%)\n", low, 100*low/total
  printf "Medium loss (1-10%%): %d (%.1f%%)\n", med, 100*med/total
  printf "High loss (>10%%): %d (%.1f%%)\n", high, 100*high/total
}'

echo ""
echo "=============================================="
echo "Latency Distribution"
echo "=============================================="

tail -n +2 training_dataset_master.csv | cut -d',' -f22 | \
awk '{
  if($1 < 20) low++
  else if($1 >= 20 && $1 < 100) med++
  else if($1 >= 100 && $1 < 500) high++
  else very_high++
}
END {
  total=low+med+high+very_high
  printf "Low (<20ms): %d (%.1f%%)\n", low, 100*low/total
  printf "Medium (20-100ms): %d (%.1f%%)\n", med, 100*med/total
  printf "High (100-500ms): %d (%.1f%%)\n", high, 100*high/total
  printf "Very high (>=500ms): %d (%.1f%%)\n", very_high, 100*very_high/total
}'

echo ""
echo "=============================================="
echo "Sample rows: Zero loss vs High loss"
echo "=============================================="

echo "ZERO LOSS samples (normal state):"
tail -n +2 training_dataset_master.csv | awk -F',' '$24 == 0' | head -2 | cut -d',' -f1-13,22-24

echo ""
echo "HIGH LOSS samples (congestion state):"
tail -n +2 training_dataset_master.csv | awk -F',' '$24 > 50' | head -2 | cut -d',' -f1-13,22-24

