function nodeTranslation = translateElu(~, nodeTranslation, node, ~)
%   Copyright 2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "alpha"       "FLOAT"      true    1
    });
% Parse the attributes
[alpha] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
Y = node.output{1};
X = node.input{1};
[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
Command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sonnxElu(%s, %f, %s);\n', varY, numDimsY, ...
    getPackageName(nodeTranslation.GraphTranslation), varX, alpha, numDimsX),...
    ];
nodeTranslation.MCode = string(Command);
nodeTranslation.IncludedFunctionNames   = "onnxElu";
end

