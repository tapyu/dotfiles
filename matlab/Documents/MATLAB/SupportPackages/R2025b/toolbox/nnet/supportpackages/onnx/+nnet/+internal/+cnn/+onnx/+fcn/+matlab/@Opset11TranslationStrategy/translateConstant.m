function nodeTranslation = translateConstant(~, nodeTranslation, node, ~)
%   Copyright 2023-2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "sparse_value"	"TENSOR"    true    []
    "value"         "TENSOR"    true    []
    });
% Parse the attributes
[sparse_value, value] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% According to ONNX Doc, exactly one of the attributes will be present.
if ~isempty(sparse_value)
    % Get the data from a SparseTensorProto
    [tensorValue, shape] = nnet.internal.cnn.onnx.getDataFromSparseTensorProto(sparse_value);
else
    % Get the data from a TensorProto
    [tensorValue, shape] = nnet.internal.cnn.onnx.getDataFromTensorProto(value);
end
% Make the value a dlarray so it could be made learnable if
% desired. Make it single because it's an activation tensor.
tensorValue = dlarray(double(tensorValue));
% Construct the constant tensor at translation-time and add it to Nonlearnables
if numel(shape)>1
    DLTShape = fliplr(shape(:)');
    tensorValue = reshape(tensorValue, DLTShape);   % Can just reshape the ONNX data because the memory ordering is the same in ONNX and DLT.
end
Y = node.output{1};
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', Y, nnet.internal.cnn.onnx.fcn.RankedArray(tensorValue, numel(shape)));
if ~isempty(value) && nnet.internal.cnn.onnx.isONNXTypeInteger(value.data_type) ||...
        ~isempty(sparse_value) && nnet.internal.cnn.onnx.isONNXTypeInteger(sparse_value.data_type)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

