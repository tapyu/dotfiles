function nodeTranslation = translateRandomUniform(~, nodeTranslation, node, ~)
%   Copyright 2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "dtype"     "INT"      true    1
    "high"      "FLOAT"    true    1.0
    "low"       "FLOAT"    true    0.0
    "seed"      "FLOAT"    true    []
    "shape"     "INTS"     false   []
    });
% Parse the attributes
[dtype, high, low, seed, shape] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% From TensorProto, dtype can be:
%     UNDEFINED = 0;
%     // Basic types.
%     FLOAT = 1;   // float
%     UINT8 = 2;   // uint8_t
%     INT8 = 3;    // int8_t
%     UINT16 = 4;  // uint16_t
%     INT16 = 5;   // int16_t
%     INT32 = 6;   // int32_t
%     INT64 = 7;   // int64_t
%     STRING = 8;  // string
%     BOOL = 9;    // bool
output   = node.output{1};
[VarOutput, NumDimOutput]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, output);

DLTShape = fliplr(shape(:)');   % Reverse-onnx shape as a row vector
ShapeStr = strjoin(string(DLTShape), ', ');
command = sprintf('%% %s:\n', node.op_type);
if ~isempty(seed)
    command = [command, sprintf('rng(%f);\n', seed)];
end
if ismember(dtype, [2 3 4 5 6 7 9])
    % Random integer or bool
    command = [command, sprintf('%s = round(rand([%s])*(%d - %d) + %d);\n', VarOutput, ShapeStr, high, low, low)];
    nodeTranslation.IntegerOutputTensorNames = string(output);
else
    % Random float
    command = [command, sprintf('%s = rand([%s])*(%d - %d) + %d;\n', VarOutput, ShapeStr, high, low, low)];
end
command = [command, sprintf('%s = %d;\n', NumDimOutput, numel(shape))];
nodeTranslation.MCode = string(command);
end