function nodeTranslation = translateTopK(~, nodeTranslation, node, IntegerTensorNames)
% Copyright 2024 The MathWorks, Inc.

% Opset 11: New attribues largest and sorted.
% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axis"      "INT"       true    -1
    "largest"   "INT"       true    1
    "sorted"    "INT"       true    1
    });
% Parse the attributes
[axis, largest, sorted] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% [T, I] = TopK(X, K)
T	= node.output{1};
I 	= node.output{2};
X	= node.input{1};
K	= node.input{2};
[VarT, NumdimT] = getVarAndNumDimName(nodeTranslation.GraphTranslation, T);
[VarI, NumdimI] = getVarAndNumDimName(nodeTranslation.GraphTranslation, I);
[VarX, NumdimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarK, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, K);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s, %s, %s] = %sonnxTopK11(%s, %s, %d, %d, %d, %s);\n', ...
    VarT, VarI, NumdimT, NumdimI, getPackageName(nodeTranslation.GraphTranslation), ...
    VarX, VarK, axis, largest, sorted, NumdimX),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "onnxTopK11";
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(T);
end
nodeTranslation.IntegerOutputTensorNames = string(I);
end