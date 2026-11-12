function nodeTranslation = translateConstant(~, nodeTranslation, node, ~)
%   Copyright 2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "sparse_value"	"SPARSE_TENSOR"    true    []
    "value"         "TENSOR"    true    []
    "value_float"   "FLOAT"     true    []
    "value_floats"  "FLOATS"    true    []
    "value_int"     "INT"       true    []
    "value_ints"    "INTS"      true    []
    "value_string"  "STRING"    true    []
    "value_strings" "STRINGS"   true    []
    });
% Parse the attributes
[sparse_value, value, ...
    value_float, value_floats, ...
    value_int, value_ints, ...
    value_string, value_strings] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);

% Get the output name
Y = node.output{1};

% According to ONNX Doc, exactly one of the attributes will be present.
tensorValue = [];
shape = [];
if ~isempty(sparse_value)
    % Get the data from a SparseTensorProto
    [tensorValue, shape] = nnet.internal.cnn.onnx.getDataFromSparseTensorProto(sparse_value);
    if nnet.internal.cnn.onnx.isONNXTypeInteger(sparse_value.values.data_type)
        nodeTranslation.IntegerOutputTensorNames = string(Y);
    end
elseif ~isempty(value)
    % Get the data from a TensorProto
    [tensorValue, shape] = nnet.internal.cnn.onnx.getDataFromTensorProto(value);
    if nnet.internal.cnn.onnx.isONNXTypeInteger(value.data_type)
        nodeTranslation.IntegerOutputTensorNames = string(Y);
    end
elseif ~isempty(value_float)
    % Get the data from the value_float attribute
    tensorValue = value_float;
    shape = [];
elseif ~isempty(value_floats)
    % Get the data from the value_floats attribute
    tensorValue = value_floats;
    shape = numel(tensorValue);
elseif ~isempty(value_int)
    % Get the data from the value_int attribute
    tensorValue = value_int;
    shape = [];
    nodeTranslation.IntegerOutputTensorNames = string(Y);
elseif ~isempty(value_ints)
    % Get the data from the value_ints attribute
    tensorValue = value_ints;
    shape = numel(tensorValue);
    nodeTranslation.IntegerOutputTensorNames = string(Y);
elseif ~isempty(value_string)
    % Get the data from the value_string attribute
    tensorValue = value_string;
    shape = [];
elseif ~isempty(value_strings)
    % Get the data from the value_strings attribute
    tensorValue = value_strings;
    shape = numel(tensorValue);
else
    % NodeTranslationError - Exactly one of the attributes must
    % be set.
    nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationError(node,...
        message('nnet_cnn_onnx:onnx:MissingConstantValue'));
end

% Make the value a dlarray so it could be made learnable if
% desired. Make it double because it's an activation tensor.
if isnumeric(tensorValue)
    tensorValue = dlarray(double(tensorValue));
end

% Construct the constant tensor at translation-time and add it to Nonlearnables
if numel(shape)>1
    DLTShape = fliplr(shape(:)');
    tensorValue = reshape(tensorValue, DLTShape);   % Can just reshape the ONNX data because the memory ordering is the same in ONNX and DLT.
end
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', Y, nnet.internal.cnn.onnx.fcn.RankedArray(tensorValue, numel(shape)));

end

