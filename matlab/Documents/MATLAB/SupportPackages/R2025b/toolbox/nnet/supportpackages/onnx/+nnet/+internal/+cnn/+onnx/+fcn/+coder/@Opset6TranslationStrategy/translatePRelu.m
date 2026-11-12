function nodeTranslation = translatePRelu(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024-2026 The MathWorks, Inc.

Y = node.output{1};
X = node.input{1};
slope	= node.input{2};
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumDimY]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
setInitializerType(nodeTranslation.GraphTranslation,'Learnables',slope);
[VarSlope, ~]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, slope);
Command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('%s = max(0,%s) + %s.*min(0,%s);\n', VarY, VarX, VarSlope, VarX),...
    sprintf('%s = %s;\n', NumDimY, NumDimX),...
    ];
nodeTranslation.MCode = string(Command);
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

