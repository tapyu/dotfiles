function nodeTranslation = translateLogSoftmax(~, nodeTranslation, node, ~)
%   Copyright 2026 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axis"         "INT"       true    -1
    });
% Parse the attributes
[axis] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% Y = LogSoftmax(X)
Y     = node.output{1};
X     = node.input{1};
[VarX, NumdimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumdimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sonnxLogSoftmax13(%s, %d, %s);\n', ...
    VarY, NumdimsY, getPackageName(nodeTranslation.GraphTranslation), VarX, axis, NumdimsX),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = ["onnxLogSoftmax13", "onnxSoftmax13", "prepareReduceArgs"];
end