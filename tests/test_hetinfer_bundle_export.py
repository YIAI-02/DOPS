"""The fast exporter retains placement, route sizes, and cached weight identity."""
from pathlib import Path
from types import SimpleNamespace as NS
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import hetinfer_experiment_export as export


def test_single_bundle_preserves_expert_projection_and_token_costs(tmp_path, monkeypatch):
    npu, pim0, pim1 = export.DEVICES
    cluster = NS(devices={name: NS(name=name, type='npu' if name==npu else 'pim',
                                   mem_capacity_GB=16, mem_bw_GBs=100)
                          for name in export.DEVICES})
    cluster.devices['CPU0'] = NS(name='CPU0', type='cpu')
    class Cost:
        def comm_cost(self, source, destination, size): return size/1000
        def pim_local_weight_load_cost(self, size, fmt, dev): return NS(total_s=.01)
        def npu_local_weight_load_cost(self, size, fmt, resident, dev): return NS(total_s=.02)
        def weight_resident_format(self, fmt, device): return fmt
        def estimate_flops(self, node, batch, sequence, phase): return 16*batch*sequence
    cost=Cost();cost.cluster=cluster
    monkeypatch.setattr(export, '_service', lambda cost,node,dev,batch,seq,phase:
                        batch*seq*({npu:2,pim0:1,pim1:3}[dev.name]))
    monkeypatch.setattr(export, 'route_time_s', lambda ctx,src,dst,size,source_layout: size/1000)
    graph=NS(nodes={})
    operators=[]
    def node(name, family, dependencies, expert=None, service=None):
        attrs={'layer_index':0,'canonical_op_slot':name.lower(),'dim':4}
        if expert is not None: attrs.update(expert=expert,expert_id=expert)
        graph.nodes[name]=NS(name=family,attrs=attrs,weight_id=name if expert else None,
                             weight_size=32 if expert else 0)
        operators.append({'op_id':'prefill:1:'+name,'dependencies':['prefill:1:'+d for d in dependencies],
            'legal_devices':[npu,pim0,pim1],'expert_device':npu,
            'service_s':service or {npu:1.,pim0:2.,pim1:3.},
            'network_metadata':{'name':family,'batch':1,'seq_len':2,'node_attrs':attrs}})
    node('Router','ROUTER',[])
    for expert in ('E0','E1'):
        node('W1_'+expert,'FFN_W1',['Router'],expert,{npu:1.,pim0:4.,pim1:6.})
        node('W2_'+expert,'FFN_W2',['W1_'+expert],expert,{npu:5.,pim0:1.,pim1:6.})
    node('Combine','COMBINE',['W2_E0','W2_E1'])
    entry={'consumer_op_id':'prefill:1:W1_E0','producer_op_id':'prefill:1:Router',
           'tensor_id':'activation','semantics':'data','bytes':16,
           'source_residencies':[{'device_id':npu,'layout':'ND'}],
           'destination_devices':[npu,pim0,pim1]}
    original={'tensor_id':'activation','source_device_id':npu,'destination_device_id':pim0,
              'bytes':16,'layout':'ND','duration_s':.25}
    snapshot={'phase':'prefill','operators':operators,'inputs':[entry],
              'routes':[original],'collective_contexts':[]}
    cfg={'model_family':'mixtral','batch':1,'prefill_len':2,'decode_len':1,
         'hetinfer_graph_id':'tiny','hetinfer_workload_id':'tp2','scheduler_seed':7}
    output=export.export_experiment_bundle(output=tmp_path/'bundle.json',cfg=cfg,snapshots=[snapshot],
        graph=graph,shape=NS(layer_num=1),cluster=cluster,cost=cost)
    bundle=json.loads(output.read_text())
    assert list(tmp_path.iterdir())==[output]
    assert 'schema' not in bundle
    exported=bundle['networks'][0]['operators']
    experts=[op for op in exported if op['expert_id']]
    assert {op['default_device'] for op in experts}=={pim0}
    assert {op['native_device'] for op in experts}=={npu}
    assert {op['weight_id'] for op in experts}=={'W1_E0','W2_E0','W1_E1','W2_E1'}
    assert all('expert_service_buckets' not in op for op in exported)
    buckets=bundle['expert_service_buckets']['prefill']['ffn_w1']
    assert [row['activation_bytes'] for row in buckets]==[8,16]
    assert [row['service_time_s'][npu] for row in buckets]==[2,4]
    assert original in bundle['movements']
    assert {'tensor_id':'activation','source_device_id':npu,'destination_device_id':pim0,
            'bytes':8,'layout':'ND','duration_s':.008} in bundle['movements']
    first=next(op for op in exported if op['op_id']=='prefill:1:W1_E0')
    assert first['inputs'][0]['tensor_id']=='weight:W1_E0'
    assert all('consumer_op_id' not in row for row in first['inputs'])
