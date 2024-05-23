git clone https://github.com/onnx/onnx
protoc -I onnx/onnx/ --python_out=. onnx/onnx/onnx-ml.proto
rmdir /s /q onnx