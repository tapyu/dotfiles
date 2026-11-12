function nodeTranslation = translateConstant(~, nodeTranslation, node, ~)
%   Copyright 2023-2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "value"      "TENSOR"    false    []
    });
% Parse the attributes
[value] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
[tensorValue, shape] = nnet.internal.cnn.onnx.getDataFromTensorProto(value);

% Make the value a dlarray so it could be made learnable. Make
% it double because it's potentially an activation tensor.
tensorValue = dlarray(double(tensorValue));
% Construct the constant tensor at translation-time and add it to
% Nonlearnables
if numel(shape)>1
    DLTShape = fliplr(shape(:)');
    tensorValue = reshape(tensorValue, DLTShape);   % Can just reshape the ONNX data because the memory ordering is the same in ONNX and DLT.
end
Y = node.output{1};
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', Y, nnet.internal.cnn.onnx.fcn.RankedArray(tensorValue, numel(shape)));

if nnet.internal.cnn.onnx.isONNXTypeInteger(value.data_type)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

