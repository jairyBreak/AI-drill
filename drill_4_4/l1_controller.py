import sys
sys.path.append('/home/p4/p4-utils')
import subprocess

from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI

class EcmpController:
    def __init__(self, switch_name):
        self.switch_name = switch_name
        self.topo = load_topo('topology.json') 
        self.thrift_port = self.topo.get_thrift_port(switch_name)

        # 建立 Thrift 連線
        self.api = SimpleSwitchThriftAPI(self.thrift_port)
        print(f"[{self.switch_name}] 已成功連線至 Thrift API (Port: {self.thrift_port})")

    def set_w_ecmp_weights(self, target_ip, weight_c1, weight_c2):
        print(f"[{self.switch_name}] 正在下發規則 -> 目標 IP: {target_ip}, 權重 C1:{weight_c1} C2:{weight_c2}")
    
        selector_name = "w_ecmp_selector"
        table_name = "w_ecmp_table"
        action_name = "assign_component"

    # 1. 建立 Group，取得 group handle (int)
    
        grp_handle = self.api.act_prof_create_group(selector_name)

    # 2. 建立 Component 1 的 Member 並加入 Group
        for _ in range(weight_c1):
        # 參考: act_prof_create_member(act_prof_name, action_name, action_params=[])
            mbr_handle = self.api.act_prof_create_member(selector_name, action_name, ["1"])
        
        # 參考: act_prof_add_member_to_group(act_prof_name, mbr_handle, grp_handle)
            self.api.act_prof_add_member_to_group(selector_name, mbr_handle, grp_handle)

    # 3. 建立 Component 2 的 Member 並加入 Group
        for _ in range(weight_c2):
            mbr_handle = self.api.act_prof_create_member(selector_name, action_name, ["2"])
            self.api.act_prof_add_member_to_group(selector_name, mbr_handle, grp_handle)

        cli_command = f'echo "table_indirect_add_with_group {table_name} {target_ip} => {grp_handle}" | simple_switch_CLI --thrift-port {self.thrift_port}'
    
        try:
            subprocess.run(cli_command, shell=True, check=True, capture_output=True)
            print(f"[{self.switch_name}] 成功將 {target_ip} 綁定至 Group {grp_handle}")
        except subprocess.CalledProcessError as e:
            print(f"CLI 執行失敗: {e.stderr.decode()}")

if __name__ == "__main__":
    # 初始化控制器
    ctrl = EcmpController("l1")
    # 測試：動態下發 1:2 的權重給 10.0.2.2
    ctrl.set_w_ecmp_weights(target_ip="10.0.2.2", weight_c1=1, weight_c2=2)