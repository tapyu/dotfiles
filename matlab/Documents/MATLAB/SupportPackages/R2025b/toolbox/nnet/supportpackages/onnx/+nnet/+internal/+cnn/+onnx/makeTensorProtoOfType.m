function tp = makeTensorProtoOfType(tensorName, tensorDims, tensorData, tensorType)
% Creates a TensorProto of the given name, dims, data, and type.

%   Copyright 2021-2026 The MathWorks, Inc.

% Make tensorData a row-major vector
tensorData = permute(tensorData, fliplr(1:ndims(tensorData)));
tensorData = tensorData(:)';

% Create TensorProto
tp = nnet.internal.cnn.onnx.TensorProto;
tp.name = tensorName;
tp.data_type = tensorType;
tp.dims = nnet.internal.cnn.onnx.dimVector(tensorDims, numel(tensorDims));

% Insert the data into the field of the corresponding type
switch tensorType
    case nnet.internal.cnn.onnx.TensorProto_DataType.FLOAT
        tp.float_data = single(tensorData);
    case nnet.internal.cnn.onnx.TensorProto_DataType.INT32
        tp.int32_data = int32(tensorData);
    case nnet.internal.cnn.onnx.TensorProto_DataType.STRING
        tp.string_data = {char(tensorData)};
    case nnet.internal.cnn.onnx.TensorProto_DataType.INT64
        tp.int64_data = int64(tensorData);
    case nnet.internal.cnn.onnx.TensorProto_DataType.DOUBLE
        tp.double_data = double(tensorData);
    case nnet.internal.cnn.onnx.TensorProto_DataType.UINT64
        tp.uint64_data = uint64(tensorData);
    case nnet.internal.cnn.onnx.TensorProto_DataType.BOOL
        tp.int32_data = int32(tensorData);
    otherwise
        tp.raw_data = nnet.internal.cnn.onnx.rawData(tensorData);
end

end