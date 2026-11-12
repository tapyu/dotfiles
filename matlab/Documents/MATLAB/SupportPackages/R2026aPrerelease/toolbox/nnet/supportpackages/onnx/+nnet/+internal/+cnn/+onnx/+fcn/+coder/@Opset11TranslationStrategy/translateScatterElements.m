function nodeTranslation = translateScatterElements(~, nodeTranslation, node, IntegerTensorNames)       % Introduced in opset 11
%   Copyright 2024 The MathWorks, Inc.

% Scatter was deprecated as of opset 11. ScatterElements-11 is
% the same as Scatter-9 except the Indices input can now have
% negative elements.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axis"      "INT"       true    0
    });
% Parse the attributes
[axis]  = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
Y       = node.output{1};
X       = node.input{1};
Indices	= node.input{2};
Updates	= node.input{3};
[VarX, NumdimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumdimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarIndices, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Indices);
[VarUpdates, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Updates);

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sonnxScatterElements(%s, %s, %s, %d, "none", %s);\n', ...
    VarY, NumdimY, getPackageName(nodeTranslation.GraphTranslation), VarX, VarIndices, VarUpdates, axis, NumdimX),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "onnxScatterElements";
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

