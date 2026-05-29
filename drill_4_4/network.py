from p4utils.mininetlib.network_API import NetworkAPI

def main():
    # 1. 初始化 API
    net = NetworkAPI()

    # --- 對應 JSON: "top-level settings" ---
    net.setLogLevel('info')
    
    # 對應 "p4_src": "p4src/ecmp.p4"
    net.setP4Source('p4src/ecmp.p4')

    # 對應 "pcap_dump": false
    net.disablePcapDumpAll()
    
    # 對應 "enable_log": false
    net.disableLogAll()

    # --- 對應 JSON: "switches" ---
    # 定義 Spine switches (s1 to s8)
    for i in range(1, 9):
        net.addP4Switch(f's{i}', cli_input=f's{i}-commands.txt')
    
    # 定義 Leaf switches (l1 to l8)
    for i in range(1, 9):
        net.addP4Switch(f'l{i}', cli_input=f'l{i}-commands.txt')

    # --- 對應 JSON: "hosts" ---
    for i in range(1, 9):
        net.addHost(f'h{i}', 
                    ip=f'10.0.{i}.{i}/24', 
                    mac=f'00:00:00:00:00:0{i}',
                    gateway=f'10.0.{i}.254')

    # --- 對應 JSON: "links" ---
    # Host 連接 (預設頻寬)
    for i in range(1, 9):
        net.addLink(f'h{i}', f'l{i}')

    # Switch 間連接
    # s1-s4 bandwidth = 1.0 (scaled to 1.0 here, matching p4app.json)
    # s5-s8 bandwidth = 1.5 (scaled to 1.5 here, matching p4app.json)
    for l in range(1, 9):
        for s in range(1, 9):
            bw = 1.5 if s >= 5 else 1.0
            net.addLink(f'l{l}', f's{s}', bw=bw, max_queue_size=256)

    # --- 對應 JSON: "topology" 策略設定 ---
    # 對應 "assignment_strategy": "mixed" 
    # net.l3() 會自動計算並填入 ARP 表項，並處理 L3 路由的基本配置
    net.l3()

    # --- 啟動網路 ---
    net.startNetwork()
    net.enableCli()

if __name__ == '__main__':
    main()