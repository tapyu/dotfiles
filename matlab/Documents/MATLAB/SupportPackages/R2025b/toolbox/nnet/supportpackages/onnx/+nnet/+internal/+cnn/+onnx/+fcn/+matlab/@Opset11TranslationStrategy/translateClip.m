function nodeTranslation = translateClip(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2026 The MathWorks, Inc.

Y       = node.output{1};
X       = node.input{1};
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
if numel(node.input) > 1 && ~isempty(node.input{2})
    MinString = node.input{2};
    [VarMinString, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, MinString);
else
    VarMinString = '-Inf';
end
if numel(node.input) > 2 && ~isempty(node.input{3})
    MaxString = node.input{3};
    [VarMaxString, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, MaxString);
else
    VarMaxString = 'Inf';
end
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('%s = min(%s, max(%s, %s));\n', VarY, VarMaxString, VarMinString, VarX),...
    sprintf('%s = %s;\n', NumDimY, NumDimX),...
    ];
nodeTranslation.MCode = string(command);
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

