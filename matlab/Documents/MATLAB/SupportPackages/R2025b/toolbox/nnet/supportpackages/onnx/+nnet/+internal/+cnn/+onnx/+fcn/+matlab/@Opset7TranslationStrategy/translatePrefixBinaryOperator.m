function nodeTranslation = translatePrefixBinaryOperator(~, nodeTranslation, node, MATLABFcn, ~)
%   Copyright 2023-2024 The MathWorks, Inc.

D       = node.output{1};
A       = node.input{1};
B       = node.input{2};
[VarA, NumDimA] = getVarAndNumDimName(nodeTranslation.GraphTranslation, A);
[VarB, NumDimB] = getVarAndNumDimName(nodeTranslation.GraphTranslation, B);
[VarD, NumDimD] = getVarAndNumDimName(nodeTranslation.GraphTranslation, D);
Command = [sprintf('%% %s:\n', node.op_type),...
    sprintf('%s = %s(%s, %s);\n', VarD, MATLABFcn, VarA, VarB),...
    sprintf('%s = max(%s, %s);\n', NumDimD, NumDimA, NumDimB),...
    ];
nodeTranslation.MCode = string(Command);
end
