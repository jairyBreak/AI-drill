/* -*- P4_16 -*- */
#include <core.p4>
#include <v1model.p4>

//My includes
#include "include/headers.p4"
#include "include/parsers.p4"

register<bit<32>>(512) q_depth_reg;
register<bit<32>>(512) last_best_p_reg;
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
        
    action drop() {
        mark_to_drop(standard_metadata);
    }

    action ecmp_group(bit<14> ecmp_group_id, bit<16> num_nhops){
        /*
        hash(meta.ecmp_hash,
	    HashAlgorithm.crc16,
	    (bit<1>)0,
	    { hdr.ipv4.srcAddr,
	      hdr.ipv4.dstAddr,
          hdr.tcp.srcPort,
          hdr.tcp.dstPort,
          hdr.ipv4.protocol},
	    num_nhops);
        */
        random(port_num_1,(bit<32>)0,(bit<32>)num_nhops-1);
        random(port_num_2,(bit<32>)0,(bit<32>)num_nhops-1);

        if(port_num_1 == port_num_2){
            if(port_num_2 == (bit<32>)(num_nhops-1)){
                port_num_2 = 0;
            }
            else{
                port_num_2 = port_num_2 + 1;
            }
        }

        port_num_1 = port_num_1 + 2;
        port_num_2 = port_num_2 + 2;
        
        last_best_p_reg.read(port_num_mem,(bit<32>)ecmp_group_id);
        if(port_num_mem < 2){
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
        if(best_queue >= queue_len_mem) {
            best_port = port_num_mem;
            best_queue = queue_len_mem;
        }

        meta.ecmp_group_id = ecmp_group_id;
        meta.ecmp_hash = (bit<14>)best_port;

        best_queue = best_queue + 1;
        q_depth_reg.write((bit<32>)best_port,best_queue);
        last_best_p_reg.write((bit<32>)ecmp_group_id,best_queue);
    }

    action set_nhop(macAddr_t dstAddr, egressSpec_t port) {
        hdr.ethernet.srcAddr = hdr.ethernet.dstAddr;
        hdr.ethernet.dstAddr = dstAddr;
        standard_metadata.egress_spec = port;
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
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
            ecmp_group;
            drop;
        }
        size = 1024;
        default_action = drop;
    }

    apply {
        if (hdr.ipv4.isValid()){
            switch (ipv4_lpm.apply().action_run){
                ecmp_group: {
                    ecmp_group_to_nhop.apply();
                }
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
        q_depth_reg.read(current_q_len,(bit<32>)standard_metadata.egress_port);
        if(current_q_len > 0){
            current_q_len = current_q_len - 1;
            q_depth_reg.write(
                (bit<32>)standard_metadata.egress_port,  // Index: 使用 Port ID 作為索引
                (bit<32>)current_q_len    // Value: 寫入當前的 Queue Depth
            );
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