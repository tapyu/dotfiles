function nodeTranslation = translateScatter(~, nodeTranslation, node, IntegerTensorNames)        % Introduced in opset 9
%   Copyright 2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axis"      "INT"       true    0
    });
% Parse the attributes
[axis] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
Y       = node.output{1};
X       = node.input{1};
Indices	= node.input{2};
Updates	= node.input{3};
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarX, NumDimX]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarIndices, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Indices);
[VarUpdates, ~]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, Updates);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sonnxScatter(%s, %s, %s, %d, %s);\n', ...
    VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, VarIndices, VarUpdates, axis, NumDimX),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "onnxScatter";
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

