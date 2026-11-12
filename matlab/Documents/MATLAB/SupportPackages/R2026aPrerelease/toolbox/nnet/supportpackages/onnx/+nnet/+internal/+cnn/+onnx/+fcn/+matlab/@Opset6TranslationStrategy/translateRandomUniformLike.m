function nodeTranslation = translateRandomUniformLike(~, nodeTranslation, node, IntegerTensorNames)        % Introduced in opset 1
%   Copyright 2025 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "dtype"         "INT"       true    []
    "high"          "FLOAT"     true    1.0
    "low"           "FLOAT"     true    0.0
    "seed"          "FLOAT"     true    []
    });
% Parse the attributes
[dtype, high, low, seed] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);

% Output = RandomUniformLike(Input, dtype, high, low, seed)
Output  = node.output{1};
Input	= node.input{1};

[VarOutput, NumDimOutput] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Output);
[VarInput, NumDimInput]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, Input);

command = sprintf('%% %s:\n', node.op_type);
if ~isempty(seed)
    command = [command, sprintf('rng(%f);\n', seed)];
end
ShapeName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Shape']);
command = [command, sprintf('%s = size(%s);\n', ShapeName, VarInput)];
if (~isempty(dtype) && ismember(dtype, [2,3,4,5,6,7,9,12,13])) || ...
        (isempty(dtype) && ismember(Input, IntegerTensorNames))
    % Random integer or bool
    command = [command, sprintf('%s = dlarray(round(rand(%s)*(%d - %d) + %d));\n', VarOutput, ShapeName, high, low, low)];
    nodeTranslation.IntegerOutputTensorNames = string(Output);
else
    % Random float
    command = [command, sprintf('%s = dlarray(rand(%s)*(%d - %d) + %d);\n', VarOutput, ShapeName, high, low, low)];
end
command = [command, sprintf('%s = %s;\n', NumDimOutput, NumDimInput)];
nodeTranslation.MCode = string(command);
end