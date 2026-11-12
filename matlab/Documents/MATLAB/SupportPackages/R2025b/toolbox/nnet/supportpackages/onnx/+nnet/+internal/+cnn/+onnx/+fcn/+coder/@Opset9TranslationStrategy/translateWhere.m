function nodeTranslation = translateWhere(~, nodeTranslation, node, IntegerTensorNames)        % Introduced in opset 9
%   Copyright 2024 The MathWorks, Inc.

output         	= node.output{1};
condition       = node.input{1};
X               = node.input{2};
Y               = node.input{3};
[VarOutput, NumDimOutput] = getVarAndNumDimName(nodeTranslation.GraphTranslation, output);
[VarCondition, NumDimCondition]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, condition);
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sonnxWhere(%s, %s, %s, %s, %s, %s);\n', ...
    VarOutput, NumDimOutput, getPackageName(nodeTranslation.GraphTranslation),... 
    VarCondition, VarX, VarY, NumDimCondition, NumDimX, NumDimY),...
    ];
nodeTranslation.MCode = string(command);
nodeTranslation.IncludedFunctionNames = "onnxWhere";
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(output);
end
end

