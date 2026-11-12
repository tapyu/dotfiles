function nodeTranslation = translateTopK(~, nodeTranslation, node, IntegerTensorNames)
% Copyright 2024 The MathWorks, Inc.

% Opset 10: K is an input.
% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axis"      "INT"       true    -1
    });
% Parse the attributes
[axis] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% [T, I] = TopK(X, K)
T	= node.output{1};
I 	= node.output{2};
X	= node.input{1};
K	= node.input{2};
[VarT, NumDimT] = getVarAndNumDimName(nodeTranslation.GraphTranslation, T);
[VarI, NumDimI] = getVarAndNumDimName(nodeTranslation.GraphTranslation, I);
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarK, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, K);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s, %s, %s] = %sonnxTopK(%s, %s, %d, %s);\n', ...
    VarT, VarI, NumDimT, NumDimI, getPackageName(nodeTranslation.GraphTranslation), VarX, VarK, axis, NumDimX),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "onnxTopK";
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(T);
end
nodeTranslation.IntegerOutputTensorNames = string(I);
end