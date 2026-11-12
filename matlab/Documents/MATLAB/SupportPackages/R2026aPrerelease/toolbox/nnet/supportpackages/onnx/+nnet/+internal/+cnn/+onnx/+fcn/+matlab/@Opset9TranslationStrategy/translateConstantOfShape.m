function nodeTranslation = translateConstantOfShape(~, nodeTranslation, node, ~)    	% Introduced in opset 9
%   Copyright 2023-2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "value"       "TENSOR"      true    []
    });
% Parse the attributes
[value] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
if isempty(value)
    % Default for optional attribute 'value' is float32 with value 0.
    val = double(0);
else
    val = double(nnet.internal.cnn.onnx.getDataFromTensorProto(value));
end
ValueName       = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Value']);
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', ValueName, nnet.internal.cnn.onnx.fcn.RankedArray(val, 0));

[VarValue, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, ValueName);

ShapeName = node.input{1};
Y         = node.output{1};
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarShape, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, ShapeName);

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sonnxConstantOfShape(%s, %s);\n', ...
    VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarValue, VarShape),...
    ];
nodeTranslation.MCode = string(command);
nodeTranslation.IncludedFunctionNames = "onnxConstantOfShape";
if isempty(value) || nnet.internal.cnn.onnx.isONNXTypeInteger(value.data_type)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

