function nodeTranslation = translateCumSum(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "exclusive" "INT"   true    0
    "reverse"   "INT"   true    0
    });
% Parse the attributes
[exclusive, reverse] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
Y       = node.output{1};
X       = node.input{1};
axis    = node.input{2};
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarAxis, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, axis);
Command = sprintf('%% %s:\n', node.op_type);
Command = [Command,...
    sprintf('[%s, %s] = %sonnxCumSum(%s, %s, %d, %d, %s);\n', ...
    VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, VarAxis, exclusive, reverse, NumDimX)];
nodeTranslation.MCode = string(Command);
nodeTranslation.IncludedFunctionNames = "onnxCumSum";
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

