function nodeTranslation = translateSqueeze(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024-2026 The MathWorks, Inc.

% Y = squeeze(X, axes)
Y       = node.output{1};
X       = node.input{1};
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
if numel(node.input) > 1
    axes = node.input{2};
    [VarAxes, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, axes);
    command = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('[%s, %s] = %sonnxSqueeze(%s, %s, %s);\n', ...
        VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, VarAxes, NumDimX),...
        ];
else
     command = [
         sprintf('%% %s:\n', node.op_type),...
         sprintf('[%s, %s] = %sonnxSqueeze(%s, [], %s);\n', ...
         VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, NumDimX),...
         ];
end
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "onnxSqueeze";
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end