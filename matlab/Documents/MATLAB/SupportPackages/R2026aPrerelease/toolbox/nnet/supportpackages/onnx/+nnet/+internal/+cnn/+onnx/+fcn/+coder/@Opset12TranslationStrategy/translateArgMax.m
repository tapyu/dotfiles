function nodeTranslation = translateArgMax(~, nodeTranslation, node, ~)
%   Copyright 2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axis"              "INT"  	true    0
    "keepdims"          "INT"  	true    1
    "select_last_index" "INT"   true    0
    });
% Parse the attributes
[axis, keepdims, select_last_index] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
Y       = node.output{1};
X       = node.input{1};
[VarX, NumdimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumdimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sonnxArgMax12(%s, %d, %d, %d, %s);\n', ...
    VarY, NumdimY, getPackageName(nodeTranslation.GraphTranslation), VarX, axis, keepdims, select_last_index, NumdimX),...
    ];
nodeTranslation.MCode                   = string(command);
if keepdims
    nodeTranslation.IncludedFunctionNames   = "onnxArgMax12";
else
    nodeTranslation.IncludedFunctionNames   = ["onnxArgMax12", "onnxSqueeze"];
end
nodeTranslation.IntegerOutputTensorNames = string(Y);
end

