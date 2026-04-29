#!/bin/bash

# ==========================================
# 系統變數設定
# ==========================================
VENV_PYTHON="$(which python3)"
P4RUN_CMD="sudo env PATH=$PATH $(which p4run)"
CONTROLLER_CMD="$VENV_PYTHON all_controller.py"
RATE_LIMITER_CMD="sudo -n env PATH=$PATH $VENV_PYTHON rate_limiter.py"

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

# 預先快取 sudo 憑證，避免背景 subshell 無 TTY 時卡住
sudo -v

# ==========================================
# 核心技巧：子殼層 (Subshell) 延遲啟動
# ==========================================
(
    sleep 10
    echo -e "\n[排程] 10 sec waiting... open all_controller.py"
    $CONTROLLER_CMD > controller_output.log 2>&1 &

    sleep 3
    echo -e "\n[排程] 正在背景套用 rate_limiter.py"
    $RATE_LIMITER_CMD > rate_limiter_output.log 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n[錯誤] rate_limiter.py 失敗，請查看 rate_limiter_output.log"
    else
        echo -e "\n[系統] ✅ 背景設施全部就緒！你可以切換到另一個終端機啟動 dataset_builder 了。"
    fi
    echo ""
) &

# ==========================================
# 霸佔前景：啟動 P4 網路
# ==========================================
echo "[2/4] 啟動 P4 網路"
echo "⚠️  若要結束實驗，請在 mininet> 輸入 exit，或按 Ctrl+D"

$P4RUN_CMD

cleanup