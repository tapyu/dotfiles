function nodeTranslation = translateSequenceAt(~, nodeTranslation, node, IntegerTensorNames)            % Introduced in opset 11
%   Copyright 2024 The MathWorks, Inc.

input_sequence  = node.input{1};
position        = node.input{2};
output          = node.output{1};
[VarInput_sequence, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, input_sequence);
[VarPosition, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, position);
[VarOutput, NumdimOutput] = getVarAndNumDimName(nodeTranslation.GraphTranslation, output);
idx1TempName = nnet.internal.cnn.onnx.fcn.uniqueCoderName('idx1');
idx2TempName = nnet.internal.cnn.onnx.fcn.uniqueCoderName('idx2');

code = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sprepareSequenceAtArgs(%s, %s);\n', ...
    idx1TempName, idx2TempName, getPackageName(nodeTranslation.GraphTranslation), VarInput_sequence, VarPosition),...
    sprintf('[%s, %s] = %s{[%s, %s]};\n', VarOutput, NumdimOutput, VarInput_sequence, idx1TempName, idx2TempName),...
    ];
nodeTranslation.MCode                   = string(code);
nodeTranslation.IncludedFunctionNames   = "prepareSequenceAtArgs";
if ismember(input_sequence, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(output);
end
end

