function nodeTranslation = translateTopK(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2024 The MathWorks, Inc.

% K is an attribute.
% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axis"      "INT"       true    -1
    "k"         "INT"       false    []
    });
% Parse the attributes
[axis, k] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% [T, I] = TopK(X)
T	= node.output{1};
I 	= node.output{2};
X	= node.input{1};
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarI, NumDimI] = getVarAndNumDimName(nodeTranslation.GraphTranslation, I);
[VarT, NumDimT] = getVarAndNumDimName(nodeTranslation.GraphTranslation, T);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s, %s, %s] = %sonnxTopK(%s, %d, %d, %s);\n', ...
    VarT, VarI, NumDimT, NumDimI, getPackageName(nodeTranslation.GraphTranslation), VarX, k, axis, NumDimX),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "onnxTopK";
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(T);
end
nodeTranslation.IntegerOutputTensorNames = string(I);
end

