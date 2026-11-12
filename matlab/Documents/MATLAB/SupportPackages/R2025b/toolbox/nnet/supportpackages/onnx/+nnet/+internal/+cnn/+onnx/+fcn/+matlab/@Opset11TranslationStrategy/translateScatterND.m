function nodeTranslation = translateScatterND(~, nodeTranslation, node, IntegerTensorNames)       % Introduced in opset 11
%   Copyright 2023-2024 The MathWorks, Inc.

Y       = node.output{1};
X       = node.input{1};
Indices	= node.input{2};
Updates	= node.input{3};
[VarX, NumdimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumdimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarIndices, NumdimIndices] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Indices);
[VarUpdates, NumdimUpdates] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Updates);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sonnxScatterND(%s, %s, %s, "none", %s, %s, %s);\n', ...
    VarY, NumdimY, getPackageName(nodeTranslation.GraphTranslation), VarX, VarIndices, VarUpdates, NumdimX, NumdimIndices, NumdimUpdates),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "onnxScatterND";
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

