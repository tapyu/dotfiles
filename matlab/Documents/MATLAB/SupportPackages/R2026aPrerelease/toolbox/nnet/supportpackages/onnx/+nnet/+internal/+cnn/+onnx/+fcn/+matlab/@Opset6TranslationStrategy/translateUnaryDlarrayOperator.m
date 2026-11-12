function nodeTranslation = translateUnaryDlarrayOperator(~, nodeTranslation, node, MATLABFcn, IntegerTensorNames)
%   Copyright 2023-2024 The MathWorks, Inc.

Y = node.output{1};
X = node.input{1};
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarX, NumDimX]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('%s = %s(dlarray(%s));\n', VarY, MATLABFcn, VarX),...
    sprintf('%s = %s;\n', NumDimY, NumDimX),...
    ];
nodeTranslation.MCode = string(command);
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

