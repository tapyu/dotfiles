function nodeTranslation = translateInfixBinaryOperator(~, nodeTranslation, node, MATLABFcn, IntegerTensorNames)
%   Copyright 2023-2026 The MathWorks, Inc.

D       = node.output{1};
A       = node.input{1};
B       = node.input{2};
[VarD, NumDimD] = getVarAndNumDimName(nodeTranslation.GraphTranslation, D);
[VarA, NumDimA] = getVarAndNumDimName(nodeTranslation.GraphTranslation, A);
[VarB, NumDimB]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, B);
Command = [sprintf('%% %s:\n', node.op_type),...
    sprintf('%s = %s %s %s;\n', VarD, VarA, MATLABFcn, VarB),...
    sprintf('%s = max(%s, %s);\n', NumDimD, NumDimA, NumDimB),...
    ];
nodeTranslation.MCode = string(Command);
if ismember(node.input{1}, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(D);
end
end

