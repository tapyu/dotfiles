function nodeTranslation = translateShape(~, nodeTranslation, node, ~)
%   Copyright 2024 The MathWorks, Inc.

Y = node.output{1};
X = node.input{1};
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sonnxShape(%s, %s, 0, %s+1);\n',...
            VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, NumDimX, NumDimX),...
           ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "onnxShape";
nodeTranslation.IntegerOutputTensorNames = string(Y);
end

