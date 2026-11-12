function nodeTranslation = translateNonZero(~, nodeTranslation, node, ~)       % Introduced in opset 9
%   Copyright 2023-2024 The MathWorks, Inc.

Y = node.output{1};
X = node.input{1};
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumDimY]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sonnxNonZero(%s, %s);\n', ...
    VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, NumDimX),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "onnxNonZero";
nodeTranslation.IntegerOutputTensorNames = string(Y);
end

