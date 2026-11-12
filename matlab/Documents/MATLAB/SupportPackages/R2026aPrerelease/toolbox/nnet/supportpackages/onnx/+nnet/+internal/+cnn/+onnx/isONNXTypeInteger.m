function tf = isONNXTypeInteger(ONNXType)
% The input can either be a member of the enumeration
% nnet.internal.cnn.onnx.TensorProto_DataType, or just the string suffix of
% those, for example 'FLOAT', 'INT32', etc. 

%   Copyright 2020 The MathWorks, Inc.    

if ischar(ONNXType)
    ONNXType = nnet.internal.cnn.onnx.TensorProto_DataType.(ONNXType);
end
tf = ismember(ONNXType, ...
    [nnet.internal.cnn.onnx.TensorProto_DataType.UINT8;
    nnet.internal.cnn.onnx.TensorProto_DataType.INT8;
    nnet.internal.cnn.onnx.TensorProto_DataType.UINT16;
    nnet.internal.cnn.onnx.TensorProto_DataType.INT16;
    nnet.internal.cnn.onnx.TensorProto_DataType.INT32;
    nnet.internal.cnn.onnx.TensorProto_DataType.INT64;
    nnet.internal.cnn.onnx.TensorProto_DataType.UINT32;
    nnet.internal.cnn.onnx.TensorProto_DataType.UINT64]);
end
