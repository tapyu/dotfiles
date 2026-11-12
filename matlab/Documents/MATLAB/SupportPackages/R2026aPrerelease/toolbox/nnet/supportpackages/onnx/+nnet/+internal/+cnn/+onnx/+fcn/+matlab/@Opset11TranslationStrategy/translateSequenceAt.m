function nodeTranslation = translateSequenceAt(~, nodeTranslation, node, IntegerTensorNames)            % Introduced in opset 11
%   Copyright 2023-2024 The MathWorks, Inc.

input_sequence  = node.input{1};
position        = node.input{2};
output          = node.output{1};
[VarInput_sequence, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, input_sequence);
[VarPosition, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, position);
[VarOutput, NumdimOutput] = getVarAndNumDimName(nodeTranslation.GraphTranslation, output);
code = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[idx1, idx2] = %sprepareSequenceAtArgs(%s, %s);\n', ...
    getPackageName(nodeTranslation.GraphTranslation), VarInput_sequence, VarPosition),...
    sprintf('[%s, %s] = %s{[idx1, idx2]};\n', VarOutput, NumdimOutput, VarInput_sequence),...
    ];
nodeTranslation.MCode                   = string(code);
nodeTranslation.IncludedFunctionNames   = "prepareSequenceAtArgs";
if ismember(input_sequence, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(output);
end
end

