/* -*- P4_16 -*- */
#include <core.p4>
#include <v1model.p4>

//My includes
#include "include/headers.p4"
#include "include/parsers.p4"

//for drill
register<bit<32>>(512) q_depth_reg;
register<bit<32>>(512) last_best_p_reg;
register<bit<32>>(1024) port_map_reg;

//for ML (maybe)
register<bit<32>>(1024) path_max_queue_depth_reg;
register<bit<32>>(1024) path_max_q_delay_reg;
register<bit<32>>(1024) path_acc_q_delay_reg;

counter(512,CounterType.packets) cnt_ingress;
counter(512,CounterType.packets) cnt_egress;
counter(512,CounterType.packets) cnt_enq;
counter(256, CounterType.packets) port_drop_counter;
counter(256, CounterType.bytes) port_bytes_counter; 

/*************************************************************************
************   C H E C K S U M    V E R I F I C A T I O N   *************
*************************************************************************/

control MyVerifyChecksum(inout headers hdr, inout metadata meta) {
    apply {  }
}

/*************************************************************************
**************  I N G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_metadata) {

    bit<32> queue_len_1;
    bit<32> queue_len_2;
    bit<32> queue_len_mem;

    bit<32> port_num_1;
    bit<32> port_num_2;
    bit<32> port_num_mem;

    bit<32> best_port;
    bit<32> best_queue;

    action set_w_ecmp(){}
    action drop() {
        mark_to_drop(standard_metadata);
    }
    action assign_component(bit<14> comp_id){
        meta.ecmp_group_id = comp_id;
    }
    action_selector(HashAlgorithm.crc16, 32, 1024) w_ecmp_selector;

    table w_ecmp_table {
        key = {
            hdr.ipv4.dstAddr : exact; 
            hdr.ipv4.srcAddr : selector;
            hdr.ipv4.protocol : selector;
            meta.l4_dstPort : selector;
            meta.l4_srcPort : selector;
        }
        actions = {
            assign_component;
            drop;
        }
        implementation = w_ecmp_selector;
        size = 1024;
    }



    action run_drill(bit<16> num_nhops){
        
        bit<32> logical_idx1;
        bit<32> logical_idx2;

        random(logical_idx1,(bit<32>)0,(bit<32>)num_nhops-1);
        random(logical_idx2,(bit<32>)0,(bit<32>)num_nhops-1);

        if(logical_idx1 == logical_idx2){
            if(logical_idx2 == (bit<32>)(num_nhops-1)){
                logical_idx2 = 0;
            }
            else{
                logical_idx2 = logical_idx2 + 1;
            }
        }

        bit<32> map_addr_1 = (bit<32>)meta.ecmp_group_id * 16 + logical_idx1;
        bit<32> map_addr_2 = (bit<32>)meta.ecmp_group_id * 16 + logical_idx2;

        port_map_reg.read(port_num_1, map_addr_1);
        port_map_reg.read(port_num_2, map_addr_2);
        last_best_p_reg.read(port_num_mem,(bit<32>)meta.ecmp_group_id);

        if(port_num_mem == 0){
            port_num_mem = port_num_1;
        }

        q_depth_reg.read(queue_len_1, port_num_1); 
        q_depth_reg.read(queue_len_2, port_num_2);
        q_depth_reg.read(queue_len_mem, port_num_mem);

        if(queue_len_1 > queue_len_2){
            best_port = port_num_2;
            best_queue = queue_len_2;
        }
        else{
            best_port = port_num_1;
            best_queue = queue_len_1;
        }
        if(best_queue > queue_len_mem) {
            best_port = port_num_mem;
            best_queue = queue_len_mem;
        }

        meta.ecmp_hash = (bit<14>)best_port;

        best_queue = best_queue + 1;
        q_depth_reg.write((bit<32>)best_port,best_queue);
        last_best_p_reg.write((bit<32>)meta.ecmp_group_id,best_port);
    }

    action set_nhop(macAddr_t dstAddr, egressSpec_t port) {
        hdr.ethernet.srcAddr = hdr.ethernet.dstAddr;
        hdr.ethernet.dstAddr = dstAddr;
        standard_metadata.egress_spec = port;
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
    }

    // 新增一個 Table，用來根據不同的 Group ID，給予不同的 DRILL 參數
    table drill_params_table {
        key = {
            meta.ecmp_group_id: exact;
        }
        actions = {
            run_drill;
            drop;
        }
        size = 1024;
    }

    table ecmp_group_to_nhop {
        key = {
            meta.ecmp_group_id:    exact;
            meta.ecmp_hash: exact;
        }
        actions = {
            drop;
            set_nhop;
        }
        size = 1024;
    }

    table ipv4_lpm {
        key = {
            hdr.ipv4.dstAddr: lpm;
        }
        actions = {
            set_nhop;
            set_w_ecmp;
            drop;
        }
        size = 1024;
        default_action = drop;
    }
apply {
        port_bytes_counter.count((bit<32>)standard_metadata.ingress_port);

        cnt_ingress.count((bit<32>)standard_metadata.ingress_port);
        
        if(hdr.tcp.isValid()){
            meta.l4_dstPort = hdr.tcp.dstPort;
            meta.l4_srcPort = hdr.tcp.srcPort;
        }
        else if(hdr.udp.isValid()){
            meta.l4_dstPort = hdr.udp.dstPort;
            meta.l4_srcPort = hdr.udp.srcPort;
        }
        else{
            meta.l4_dstPort = 0;
            meta.l4_srcPort = 0;
        }

        if (hdr.ipv4.isValid()) {
            switch (ipv4_lpm.apply().action_run) {
                set_w_ecmp: {
                    // Selector 依 5-tuple 抽出 Component ID
                    if (w_ecmp_table.apply().hit) {
                        //  Component ID 給予對應的 DRILL 參數
                        if (drill_params_table.apply().hit) {
                            // 將 DRILL 算出的 best_port 轉換為實體 Port 與 MAC
                            ecmp_group_to_nhop.apply();
                        }
                    }
                }
                set_nhop: {
                    // 已經在 set_nhop 寫好 egress_specｓ
                }
            }
        }

        bit<32> local_q_depth = 0;
        bit<32> src_add = 0;
        if(standard_metadata.egress_spec < 255){
            cnt_enq.count((bit<32>)standard_metadata.egress_spec);
            q_depth_reg.read(local_q_depth,(bit<32>)standard_metadata.egress_spec);
        }

        if(hdr.int_hdr.isValid()){
            if(local_q_depth > hdr.int_hdr.path_queue_depth){
                hdr.int_hdr.path_queue_depth = local_q_depth;
            }
            if(standard_metadata.egress_spec == 1){
                src_add = (bit<32>)hdr.int_hdr.src_id * 16 + (bit<32>)standard_metadata.ingress_port;
                bit<32> current_depth = 0;
                path_max_queue_depth_reg.read(current_depth, (bit<32>)src_add);
                if(hdr.int_hdr.path_queue_depth > current_depth){                  
                    path_max_queue_depth_reg.write((bit<32>)src_add, hdr.int_hdr.path_queue_depth);
                }
                
                bit<32> current_max_delay = 0;
                path_max_q_delay_reg.read(current_max_delay, (bit<32>)src_add);
                if(hdr.int_hdr.max_q_delay > current_max_delay){                  
                    path_max_q_delay_reg.write((bit<32>)src_add, hdr.int_hdr.max_q_delay);
                }
                
                // For accumulated delay, we also want the maximum in the time window
                bit<32> current_acc_delay = 0;
                path_acc_q_delay_reg.read(current_acc_delay, (bit<32>)src_add);
                if(hdr.int_hdr.acc_q_delay > current_acc_delay){
                    path_acc_q_delay_reg.write((bit<32>)src_add, hdr.int_hdr.acc_q_delay);
                }
                
                hdr.ethernet.etherType = hdr.int_hdr.next_proto;
                hdr.int_hdr.setInvalid();
            }
        }
        else{
            if(hdr.ipv4.isValid() && standard_metadata.ingress_port == 1){
                hdr.int_hdr.setValid();
                hdr.int_hdr.path_queue_depth = local_q_depth;
                hdr.int_hdr.max_q_delay = 0;
                hdr.int_hdr.acc_q_delay = 0;
                hdr.int_hdr.src_id = (bit<16>)hdr.ipv4.srcAddr[7:0];
                hdr.int_hdr.next_proto = hdr.ethernet.etherType;
                hdr.ethernet.etherType = 0x9999;
            }
        }
    }
}

/*************************************************************************
****************  E G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {
    
    bit<32> current_q_len;
    apply {
        current_q_len = (bit<32>)standard_metadata.deq_qdepth;
        q_depth_reg.write(
            (bit<32>)standard_metadata.egress_port,  // Index
            (bit<32>)current_q_len    // Value
        );
        if((bit<32>)standard_metadata.egress_port < 512){
            cnt_egress.count((bit<32>)standard_metadata.egress_port);
        }
        
        if (hdr.int_hdr.isValid()) {
            bit<32> q_delay = (bit<32>)standard_metadata.deq_timedelta;
            hdr.int_hdr.acc_q_delay = hdr.int_hdr.acc_q_delay + q_delay;
            if (q_delay > hdr.int_hdr.max_q_delay) {
                hdr.int_hdr.max_q_delay = q_delay;
            }
        }
    }
}

/*************************************************************************
*************   C H E C K S U M    C O M P U T A T I O N   **************
*************************************************************************/

control MyComputeChecksum(inout headers hdr, inout metadata meta) {
     apply {
	update_checksum(
	    hdr.ipv4.isValid(),
            { hdr.ipv4.version,
	          hdr.ipv4.ihl,
              hdr.ipv4.dscp,
              hdr.ipv4.ecn,
              hdr.ipv4.totalLen,
              hdr.ipv4.identification,
              hdr.ipv4.flags,
              hdr.ipv4.fragOffset,
              hdr.ipv4.ttl,
              hdr.ipv4.protocol,
              hdr.ipv4.srcAddr,
              hdr.ipv4.dstAddr },
              hdr.ipv4.hdrChecksum,
              HashAlgorithm.csum16);
    }
}

/*************************************************************************
***********************  S W I T C H  *******************************
*************************************************************************/

//switch architecture
V1Switch(
MyParser(),
MyVerifyChecksum(),
MyIngress(),
MyEgress(),
MyComputeChecksum(),
MyDeparser()
) main;

