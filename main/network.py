from p4utils.mininetlib.network_API import NetworkAPI

def main():
    net = NetworkAPI()

    net.setLogLevel('info')
    net.setP4Source('p4src/ecmp.p4')
    net.disablePcapDumpAll()
    net.disableLogAll()

    # Spine switches s1..s8
    for i in range(1, 9):
        net.addP4Switch(f's{i}', cli_input=f's{i}-commands.txt')

    # Leaf switches l1..l8
    for i in range(1, 9):
        net.addP4Switch(f'l{i}', cli_input=f'l{i}-commands.txt')

    # hosts
    for i in range(1, 9):
        net.addHost(f'h{i}', 
                    ip=f'10.0.{i}.{i}/24', 
                    mac=f'00:00:00:00:00:0{i}',
                    gateway=f'10.0.{i}.254')

    # host links (default bw)
    for i in range(1, 9):
        net.addLink(f'h{i}', f'l{i}')

    # leaf-spine links: s1,s2=0.6 | s3,s4=0.8 | s5,s6=1.0 | s7,s8=1.2
    for l in range(1, 9):
        for s in range(1, 9):
            if s in [1, 2]:
                bw = 0.6
            elif s in [3, 4]:
                bw = 0.8
            elif s in [5, 6]:
                bw = 1.0
            else:
                bw = 1.2
            net.addLink(f'l{l}', f's{s}', bw=bw, max_queue_size=256)

    # l3() auto-fills ARP entries and basic L3 routing
    net.l3()

    net.startNetwork()
    net.enableCli()

if __name__ == '__main__':
    main()