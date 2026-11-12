function nodeTranslation = translateMatMul(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024-2026 The MathWorks, Inc.

% D = MatMul(A,B)
D = node.output{1};
A = node.input{1};
B = node.input{2};
setInitializerType(nodeTranslation.GraphTranslation,'Learnables',A);
setInitializerType(nodeTranslation.GraphTranslation,'Learnables',B);
[VarA, NumDimA] = getVarAndNumDimName(nodeTranslation.GraphTranslation, A);
[VarB, NumDimB]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, B);
[VarD, NumDimD]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, D);

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sonnxMatMul(%s, %s, %s, %s);\n', VarD, NumDimD, ...
    getPackageName(nodeTranslation.GraphTranslation), VarA, VarB, NumDimA, NumDimB),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "onnxMatMul";
if ismember(node.input{1}, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(D);
end
end

