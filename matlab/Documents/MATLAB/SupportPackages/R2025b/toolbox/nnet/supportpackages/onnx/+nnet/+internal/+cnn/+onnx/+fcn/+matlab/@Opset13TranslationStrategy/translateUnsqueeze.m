function nodeTranslation = translateUnsqueeze(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2026 The MathWorks, Inc.

% Y = unsqueeze(X, axes)
Y       = node.output{1};
X       = node.input{1};
axes    = node.input{2};
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarAxes, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, axes);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[shape, %s] = %sprepareUnsqueezeArgs(%s, %s, %s);\n', ...
    NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, VarAxes, NumDimX),...
    sprintf('%s = reshape(%s, shape);\n', VarY, VarX),...
    ];
nodeTranslation.MCode = string(command);
nodeTranslation.IncludedFunctionNames = "prepareUnsqueezeArgs";
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end