function nodeTranslation = translateArgMax(~, nodeTranslation, node, ~)
%   Copyright 2023-2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axis"     	"INT"  	true    0
    "keepdims"	"INT"  	true    1
    });
% Parse the attributes
[axis, keepdims] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
Y       = node.output{1};
X       = node.input{1};
[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sonnxArgMax(%s, %d, %d, %s);\n', varY, numDimsY, ...
    getPackageName(nodeTranslation.GraphTranslation), varX, axis, keepdims, numDimsX),...
    ];
nodeTranslation.MCode                   = string(command);
if keepdims
    nodeTranslation.IncludedFunctionNames   = "onnxArgMax";
else
    nodeTranslation.IncludedFunctionNames   = ["onnxArgMax", "onnxSqueeze"];
end
nodeTranslation.IntegerOutputTensorNames = string(Y);
end