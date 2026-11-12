function [LinearData, ONNXShape] = getDataFromTensorProto(tproto)

%   Copyright 2021 The MathWorks, Inc.

ONNXShape = tproto.dims;
switch tproto.data_type
    % Types with dedicated fields
    case nnet.internal.cnn.onnx.TensorProto_DataType.FLOAT
        LinearData = typecast(getDataFromTypedFieldOrRawData(tproto, 'float_data'), 'single');    % ONNX IR spec says "float" is a 32-bit value.
    case nnet.internal.cnn.onnx.TensorProto_DataType.INT32
        LinearData = typecast(getDataFromTypedFieldOrRawData(tproto, 'int32_data'), 'int32');
    case nnet.internal.cnn.onnx.TensorProto_DataType.STRING
        LinearData = string(getDataFromTypedFieldOrRawData(tproto, 'string_data'));
    case nnet.internal.cnn.onnx.TensorProto_DataType.INT64
        LinearData = typecast(getDataFromTypedFieldOrRawData(tproto, 'int64_data'), 'int64');
    case nnet.internal.cnn.onnx.TensorProto_DataType.DOUBLE
        LinearData = typecast(getDataFromTypedFieldOrRawData(tproto, 'double_data'), 'double');
    case nnet.internal.cnn.onnx.TensorProto_DataType.UINT64
        LinearData = typecast(getDataFromTypedFieldOrRawData(tproto, 'uint64_data'), 'uint64');
    case nnet.internal.cnn.onnx.TensorProto_DataType.BOOL
        LinearData = cast(getDataFromTypedFieldOrRawData(tproto, 'int32_data'), 'logical');
        % Other types
    case nnet.internal.cnn.onnx.TensorProto_DataType.UINT8
        LinearData = typecast(getDataFromTypedFieldOrRawData(tproto, 'raw_data'), 'uint8');
    case nnet.internal.cnn.onnx.TensorProto_DataType.INT8
        LinearData = typecast(getDataFromTypedFieldOrRawData(tproto, 'raw_data'), 'int8');
    case nnet.internal.cnn.onnx.TensorProto_DataType.UINT16
        LinearData = typecast(getDataFromTypedFieldOrRawData(tproto, 'raw_data'), 'uint16');
    case nnet.internal.cnn.onnx.TensorProto_DataType.INT16
        LinearData = typecast(getDataFromTypedFieldOrRawData(tproto, 'raw_data'), 'int16');
    case nnet.internal.cnn.onnx.TensorProto_DataType.UINT32
        LinearData = typecast(getDataFromTypedFieldOrRawData(tproto, 'raw_data'), 'uint32');
    otherwise
        error(message('nnet_cnn_onnx:onnx:InitializerDataTypeUnsupported', tproto.name,...
            char(tproto.data_type)))
end
end


function data = getDataFromTypedFieldOrRawData(tproto, fieldName)
% Look in the field name and also in 'raw_data'
if ~isempty(tproto.(fieldName))
    data = tproto.(fieldName);
elseif ~isempty(tproto.raw_data)
    data = tproto.raw_data;
else
    data = [];
end
end