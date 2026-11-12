function nodeTranslation = translateIdentity(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2024 The MathWorks, Inc.

Y = node.output{1};
X = node.input{1};
[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
Command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('%s = %s;\n', varY, varX),...
    sprintf('%s = %s;\n', numDimsY, numDimsX),...
    ];
nodeTranslation.MCode = string(Command);
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

