function nodeTranslation = translateSize(~, nodeTranslation, node, ~)
%   Copyright 2024 The MathWorks, Inc.

Y = node.output{1};
X = node.input{1};
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarX, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('%s = numel(%s);\n',VarY, VarX),...
    sprintf('%s = 1;\n', NumDimY)];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IntegerOutputTensorNames = string(Y);
end