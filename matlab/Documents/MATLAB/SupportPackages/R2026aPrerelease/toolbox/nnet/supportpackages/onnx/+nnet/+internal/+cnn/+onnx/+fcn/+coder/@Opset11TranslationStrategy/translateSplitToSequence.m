function nodeTranslation = translateSplitToSequence(~, nodeTranslation, node, IntegerTensorNames)       % Introduced in opset 11
%   Copyright 2024-2025 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axis"      "INT"       true    0
    "keepdims"  "INT"       true    1
    });
% Parse the attributes
[axis, keepdims] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
input = node.input{1};
output_sequence = node.output{1};
[VarInput, NumdimInput] = getVarAndNumDimName(nodeTranslation.GraphTranslation, input);
[VarOutput_sequence, NumdimOutput_sequence] = getVarAndNumDimName(nodeTranslation.GraphTranslation, output_sequence);
if numel(node.input)>1 && ~isempty(node.input{2})
    split = node.input{2};
    [splitStr, splitNumDimsStr] = getVarAndNumDimName(nodeTranslation.GraphTranslation, split);
else
    splitStr = '[]';
    splitNumDimsStr = '0';
end
code = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sonnxSplitToSequence(%s, %d, %s, %d, %s, %s);\n',...
    VarOutput_sequence, NumdimOutput_sequence, getPackageName(nodeTranslation.GraphTranslation), ...
    VarInput, axis, splitStr, keepdims, NumdimInput, splitNumDimsStr),...
    ];
nodeTranslation.MCode                   = string(code);
nodeTranslation.IncludedFunctionNames   = ["onnxSplitToSequence", "onnxSqueeze"];
if ismember(input, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(output_sequence);
end
end

