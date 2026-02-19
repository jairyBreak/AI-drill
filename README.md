# 石石專題
## topology
- bandwidth : 0.5 Mbps
- link queue_depth : 256
## TODO
### P4
- 固定間隔檢查並清理 queue_register for maximum queue depth.
- [v] for group packet drop rate: 增加 counter 計算 ingress/egress 時的封包數
- collect packet number and timestamp for link utilization and jitter.
### control plane
-  W-ECMP 的分組 跟 Match-Action Table 的下放
-  修改 topology.json 建立非對稱拓樸
-  Thrift API 讀取 Switch register 與 counter 狀態。
-  透過 controller 動態寫入 W-ECMP 權重參數
### ML
- 製作訓練腳本，收集資料
- 編寫模型然後訓練 (之類的)
- 線性調整 DRILL 的 d (之類的)
## Note
1. The packet drop is undetectable when the load is too large for bandwidth. (except iperf and netstat -i)  
No need to test the drop rate in this situation since data center don't act like that. Just meature the drop rate for the loss of link. (count in P4 and calculate in controller)
2. the information for W-ECMP training (still need P4 register or counter): group packet loss rate, group link utilization, Maximum jitter and queue_deqth in group, global link utilizationj Standard Deviation.