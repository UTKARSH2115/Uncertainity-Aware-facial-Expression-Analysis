import numpy as np, onnx
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType
from m3.edge import onnx_tools as O
from m3.utils.common import load_npz
from m3.infer import predict as P
rid="mnv3s-soft-s9"; D=load_npz(); idx=P.split_idx(D)["valA"][:600]; y=D["y_maj"][idx]; imgs=D["img"][idx]
d=O.onnx_dir(rid); acc=lambda p: (O.ort_logits(O.session(p,4),imgs,"mnv3s",64).argmax(1)==y).mean()
print("fp32",acc(d/"fp32.onnx"),"pre",acc(d/"fp32_pre.onnx"))
m=onnx.load(str(d/"fp32_pre.onnx")); convs=[n for n in m.graph.node if n.op_type=="Conv"]
print(len(convs), [n.name for n in convs][:5], [n.name for n in m.graph.node if n.op_type in("Gemm","MatMul")])
x=O.preprocess_np(O.calib_images(),"mnv3s",64)
def q(name,**kw):
    out=d/f"probe_{name}.onnx"
    quantize_static(str(d/"fp32_pre.onnx"),str(out),O.Reader(x),quant_format=QuantFormat.QDQ,per_channel=True,weight_type=QuantType.QInt8,activation_type=kw.pop("act",QuantType.QUInt8),op_types_to_quantize=kw.pop("ops",O.OPS),**kw)
    print(name,acc(out),flush=True)



import onnx
gem=[n.name for n in m.graph.node if n.op_type in("Gemm","MatMul")]
def group(n): return [a.i for a in n.attribute if a.name=="group"][0]
dw=[n.name for n in convs if group(n)>1]
q("excl_first",nodes_to_exclude=[convs[0].name])
q("excl_first_dw",nodes_to_exclude=[convs[0].name]+dw)
q("excl_first_dw_se",nodes_to_exclude=[convs[0].name]+dw+[n.name for n in convs if "/fc" in n.name])
