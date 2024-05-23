import onnx
onnx.checker.check_model("test.onnx")
from onnxruntime import InferenceSession
session = InferenceSession("test.onnx")
session.run([],{})
