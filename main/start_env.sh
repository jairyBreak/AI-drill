#!/bin/bash

# ---- system variables ----
VENV_PYTHON="$(which python3)"
P4RUN_CMD="sudo env PATH=$PATH $(which p4run)"
CONTROLLER_CMD="$VENV_PYTHON all_controller.py"
ML_DASHBOARD_CMD="$VENV_PYTHON realtime_ml_controller.py --web-host ${DASHBOARD_HOST:-127.0.0.1} --web-port ${DASHBOARD_PORT:-8080}"
RATE_LIMITER_CMD="sudo -n env PATH=$PATH $VENV_PYTHON rate_limiter.py"
DASHBOARD_ENABLE="${DASHBOARD_ENABLE:-1}"

# ---- shutdown / cleanup ----
cleanup() {
    echo -e "\n[系統] 偵測到關閉訊號，開始清理環境"

    echo "[清理] 關閉背景的 SDN 控制器"
    sudo pkill -f all_controller.py
    sudo pkill -f realtime_ml_controller.py

    echo "[清理] 執行 Mininet 強制清除"
    sudo mn -c > /dev/null 2>&1

    echo "[系統] 環境安全退出。"
    exit 0
}

trap cleanup SIGINT SIGTERM

# ---- main startup ----
echo "=== 啟動 SDN 自動化環境 ==="

echo "[1/4] 清理潛在的舊環境"
sudo mn -c > /dev/null 2>&1

# cache sudo credentials so the background subshell doesn't block on a TTY prompt
sudo -v

# delayed-start subshell
(
    sleep 5
    echo -e "\n[排程] 等待所有交換機 Thrift 端口就緒..."
    for port in $(seq 9090 9105); do
        until nc -z 127.0.0.1 $port 2>/dev/null; do sleep 1; done
    done

    echo -e "\n[排程] 正在背景套用 rate_limiter.py"
    $RATE_LIMITER_CMD > rate_limiter_output.log 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n[錯誤] rate_limiter.py 失敗，請查看 rate_limiter_output.log"
    else
        if [ "$DASHBOARD_ENABLE" = "0" ]; then
            echo -e "\n[排程] 啟動 all_controller.py"
            $CONTROLLER_CMD > controller_output.log 2>&1 &
            echo -e "\n[系統] 背景設施全部就緒。"
        else
            echo -e "\n[排程] 啟動 realtime_ml_controller.py dashboard"
            $ML_DASHBOARD_CMD > controller_output.log 2>&1 &
            echo -e "\n[系統] 背景設施全部就緒。Dashboard: http://${DASHBOARD_HOST:-127.0.0.1}:${DASHBOARD_PORT:-8080}"
        fi
    fi
    echo ""
) &

# foreground: start the P4 network
echo "[2/4] 啟動 P4 網路"
echo "⚠️  若要結束實驗，請在 mininet> 輸入 exit，或按 Ctrl+D"

$P4RUN_CMD

cleanup
