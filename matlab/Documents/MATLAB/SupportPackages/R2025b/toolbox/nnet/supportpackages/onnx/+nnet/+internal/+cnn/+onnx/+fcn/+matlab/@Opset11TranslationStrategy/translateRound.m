function nodeTranslation = translateRound(~, nodeTranslation, node, ~)                	% Introduced in opset 11
%   Copyright 2023-2024 The MathWorks, Inc.

Y       = node.output{1};
X       = node.input{1};
[VarX, NumdimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumdimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sonnxRound(%s, %s);\n', VarY, NumdimY, getPackageName(nodeTranslation.GraphTranslation), VarX, NumdimX),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "onnxRound";
end

