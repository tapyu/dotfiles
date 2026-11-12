function nodeTranslation = translateUnaryOperator(~, nodeTranslation, node, MATLABFcn, IntegerTensorNames)
%   Copyright 2023-2024 The MathWorks, Inc.

Y = node.output{1};
X = node.input{1};
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('%s = %s(%s);\n', VarY, MATLABFcn, VarX),...
    sprintf('%s = %s;\n', NumDimY, NumDimX),...
    ];
nodeTranslation.MCode = string(command);
if ismember(node.input{1}, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

