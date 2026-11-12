function [tensorValue, tensorShape, tensorType, translationIssue] = getDataFromConstantNode(node)
% Extracts tensor value from the attributes of a Constant operator.

%   Copyright 2020-2021 The MathWorks, Inc.

% Parse the attributes
AttributeTable = cell2table({
    "sparse_value"	"SPARSE_TENSOR"     true    []
    "value"         "TENSOR"            true    []
    "value_float"   "FLOAT"             true    []
    "value_floats"  "FLOATS"            true    []
    "value_int"     "INT"               true    []
    "value_ints"    "INTS"              true    []
    "value_string"  "STRING"            true    []
    "value_strings" "STRINGS"           true    []
    });
[sparse_value, value, ...
    value_float, value_floats, ...
    value_int, value_ints, ...
    value_string, value_strings] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);

translationIssue = nnet.internal.cnn.onnx.NodeTranslationIssue.empty;
tensorValue = [];
tensorShape = [];
tensorType = [];
% According to ONNX Doc, exactly one of the attributes will be present.
if ~isempty(sparse_value)
    % Get the data from a SparseTensorProto
    [tensorValue, tensorShape] = nnet.internal.cnn.onnx.getDataFromSparseTensorProto(sparse_value);
    tensorType = sparse_value.values.data_type;
elseif ~isempty(value)
    % Get the data from a TensorProto
    [tensorValue, tensorShape] = nnet.internal.cnn.onnx.getDataFromTensorProto(value);
    tensorType = value.data_type;
elseif ~isempty(value_float)
    % Get the data from the value_float attribute
    tensorValue = value_float;
    tensorShape = [];
    tensorType = nnet.internal.cnn.onnx.TensorProto_DataType.FLOAT;    
elseif ~isempty(value_floats)
    % Get the data from the value_floats attribute
    tensorValue = value_floats;
    tensorShape = numel(tensorValue);
    tensorType = nnet.internal.cnn.onnx.TensorProto_DataType.FLOAT;
elseif ~isempty(value_int)
    % Get the data from the value_int attribute
    tensorValue = value_int;
    tensorShape = [];
    tensorType = nnet.internal.cnn.onnx.TensorProto_DataType.INT64;
elseif ~isempty(value_ints)
    % Get the data from the value_ints attribute
    tensorValue = value_ints;
    tensorShape = numel(tensorValue);
    tensorType = nnet.internal.cnn.onnx.TensorProto_DataType.INT64;    
elseif ~isempty(value_string)
    % Get the data from the value_string attribute
    tensorValue = value_string;
    tensorShape = [];
    tensorType = nnet.internal.cnn.onnx.TensorProto_DataType.STRING;    
elseif ~isempty(value_strings)
    % Get the data from the value_strings attribute
    tensorValue = value_strings;
    tensorShape = numel(tensorValue);
    tensorType = nnet.internal.cnn.onnx.TensorProto_DataType.STRING;        
else
    % NodeTranslationError - Exactly one of the attributes must
    % be set.
    translationIssue = nnet.internal.cnn.onnx.NodeTranslationError(node,...
            message('nnet_cnn_onnx:onnx:MissingConstantValue'));
end
end