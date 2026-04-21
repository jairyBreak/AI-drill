#!/bin/bash

# ==========================================
# 系統變數設定
# ==========================================
PYTHON_CMD="/home/p4/src/p4dev-python-venv/bin/python"
P4RUN_CMD="sudo /home/linuxey/Capstone/p4dev-python-venv/bin/p4run"
CONTROLLER_CMD="$PYTHON_CMD all_controller.py"
RATE_LIMITER_CMD="sudo $PYTHON_CMD rate_limiter.py"

# ==========================================
# 終止與清理機制
# ==========================================
cleanup() {
    echo -e "\n[系統] 偵測到關閉訊號，開始清理環境"
    
    echo "[清理] 關閉背景的 SDN 控制器"
    sudo pkill -f all_controller.py
    
    echo "[清理] 執行 Mininet 強制清除"
    sudo mn -c > /dev/null 2>&1
    
    echo "[系統] 環境安全退出。"
    exit 0
}

# 綁定 SIGINT (Ctrl+C)
trap cleanup SIGINT SIGTERM

# ==========================================
# 主啟動流程
# ==========================================
echo "=== 啟動 SDN 自動化環境 ==="

echo "[1/4] 清理潛在的舊環境"
sudo mn -c > /dev/null 2>&1

# ==========================================
# 核心技巧：子殼層 (Subshell) 延遲啟動
# ==========================================
# 我們把 Python 控制器的啟動指令包成一個時間排程，丟到背景默默等待
(
    sleep 10
    echo -e "\n[排程] 10 sec waiting... open all_controller.py"
    $CONTROLLER_CMD > controller_output.log 2>&1 &
    
    sleep 3
    echo -e "\n[排程] 正在背景套用 rate_limiter.py"
    $RATE_LIMITER_CMD > rate_limiter_output.log 2>&1
    echo -e "\n[系統] ✅ 背景設施全部就緒！你可以切換到另一個終端機啟動 dataset_builder 了。"
    # 稍微敲一個 Enter 讓 mininet 的 prompt 保持乾淨
    echo "" 
) &

# ==========================================
# 霸佔前景：啟動 P4 網路
# ==========================================
echo "[2/4] 啟動 P4 網路"
echo "⚠️  若要結束實驗，請在 mininet> 輸入 exit，或按 Ctrl+D"

$P4RUN_CMD

cleanup