function nodeTranslation = translateSoftplus(this, nodeTranslation, node, ~)
%   Copyright 2025 The MathWorks, Inc.

Y       = node.output{1};
X       = node.input{1};

[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);

Command = [sprintf('%% %s:\n', node.op_type),...
    sprintf('%s = max(%s, 0) + log(1 + exp(-abs(%s)));\n', VarY, VarX, VarX),...
    sprintf('%s = %s;\n',NumDimY, NumDimX);
    ];
nodeTranslation.MCode = string(Command);
end

