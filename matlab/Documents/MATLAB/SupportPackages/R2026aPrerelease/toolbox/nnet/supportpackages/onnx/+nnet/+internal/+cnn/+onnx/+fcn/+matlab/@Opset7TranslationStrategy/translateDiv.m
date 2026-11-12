function nodeTranslation = translateDiv(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2026 The MathWorks, Inc.

% This is a binary infix operator, but we're not calling
% translateInfixBinaryOperator because we need to generate
% different code when doing integer division.
D       = node.output{1};
A       = node.input{1};
B       = node.input{2};
[VarD, NumDimD] = getVarAndNumDimName(nodeTranslation.GraphTranslation, D);
[VarA, NumDimA] = getVarAndNumDimName(nodeTranslation.GraphTranslation, A);
[VarB, NumDimB]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, B);
if ismember(node.input{1}, IntegerTensorNames)
    divCommand = sprintf('%s = fix(%s ./ %s);\n', VarD, VarA, VarB);      % integer division
    nodeTranslation.IntegerOutputTensorNames = string(D);
else
    divCommand = sprintf('%s = %s ./ %s;\n', VarD, VarA, VarB);           % float division
end
Command = [...
    sprintf('%% %s:\n', node.op_type),...
    divCommand,...
    sprintf('%s = max(%s, %s);\n', NumDimD, NumDimA, NumDimB),...
    ];
nodeTranslation.MCode = string(Command);
end

