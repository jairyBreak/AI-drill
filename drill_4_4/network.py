from p4utils.mininetlib.network_API import NetworkAPI

def main():
    # 1. 初始化 API
    net = NetworkAPI()

    # --- 對應 JSON: "top-level settings" ---
    net.setLogLevel('info')
    
    # 對應 "p4_src": "p4src/ecmp.p4"
    # 注意：這裡假設您的 P4 程式碼在相對路徑 p4src/ecmp.p4
    net.setP4Source('p4src/ecmp.p4')

    # 對應 "pcap_dump": true
    net.enablePcapDumpAll()
    
    # 對應 "enable_log": true
    net.enableLogAll()

    # --- 對應 JSON: "switches" ---
    # 定義 Spine switches (s1, s2)
    net.addP4Switch('s1', cli_input='s1-commands.txt')
    net.addP4Switch('s2', cli_input='s2-commands.txt')
    
    # 定義 Leaf switches (l1, l2)
    net.addP4Switch('l1', cli_input='l1-commands.txt')
    net.addP4Switch('l2', cli_input='l2-commands.txt')

    # --- 對應 JSON: "hosts" ---
    # NetworkAPI 允許直接在 addHost 中指定 ip, mac 和 gateway
    
    # Host 1
    net.addHost('h1', 
                ip='10.0.1.1/24', 
                mac='00:00:00:00:00:01',
                gateway='10.0.1.254') # 對應 JSON 中的 "gw"
    
    # Host 2
    net.addHost('h2', 
                ip='10.0.2.2/24', 
                mac='00:00:00:00:00:02',
                gateway='10.0.2.254')

    # --- 對應 JSON: "links" ---
    # Host 連接 (預設頻寬)
    net.addLink('h1', 'l1')
    net.addLink('h2', 'l2')

    # Switch 間連接 (設定 bw: 5)
    # L1 連接到 Spines
    net.addLink('l1', 's1', bw=5)
    net.addLink('l1', 's2', bw=5)
    
    # L2 連接到 Spines
    net.addLink('l2', 's1', bw=5)
    net.addLink('l2', 's2', bw=5)

    # --- 對應 JSON: "topology" 策略設定 ---
    # 對應 "assignment_strategy": "mixed" 
    # 雖然您手動指定了 IP，但 mixed 策略允許混合自動與手動。
    # 實際上，由於我們已經手動指定了 Host IP，這裡主要需要的是自動 ARP 功能。
    
    # 對應 "auto_arp_tables": true 與 "auto_gw_arp": true
    # net.l3() 會自動計算並填入 ARP 表項，並處理 L3 路由的基本配置
    net.l3()

    # --- 啟動網路 ---
    # 對應 "cli": true
    # startNetwork() 啟動後，我們顯式呼叫 enableCli()
    net.startNetwork()
    net.enableCli()

if __name__ == '__main__':
    main()