function nodeTranslation = translateRange(~, nodeTranslation, node, IntegerTensorNames)                 % Introduced in opset 11
%   Copyright 2024-2025 The MathWorks, Inc.

% Range is a function, not a primitive operator
output	= node.output{1};
start	= node.input{1};
limit	= node.input{2};
delta  	= node.input{3};
[VarOutput, NumDimOutput] = getVarAndNumDimName(nodeTranslation.GraphTranslation, output);
[VarStart, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, start);
[VarLimit, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, limit);
[VarDelta, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, delta);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('%s = dlarray(%s:%s:%s-sign(%s))'';\n', VarOutput, VarStart, VarDelta, VarLimit, VarDelta),...
    sprintf('%s = 1;\n', NumDimOutput),...
    ];
nodeTranslation.MCode = string(command);
if ismember(start, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(output);
end
end

