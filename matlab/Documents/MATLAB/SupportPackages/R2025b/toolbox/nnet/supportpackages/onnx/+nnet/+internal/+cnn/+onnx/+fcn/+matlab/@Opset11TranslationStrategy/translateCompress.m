function nodeTranslation = translateCompress(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axis"         "INT"       true    []
    });
% Parse the attributes
[axis] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
Y       = node.output{1};
X       = node.input{1};
condition = node.input{2};
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarCondition, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, condition);

if isempty(axis)
    % X is implicitly flattened first.
    command = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('[Indices, %s] = %sprepareCompressArgs11(%s, %s, [], %s);\n', ...
        NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, VarCondition, NumDimX),...
        sprintf('%s = %s(Indices);\n', VarY, VarX),...
        ];
else
    % X is not flattened first.
    command = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('[Indices, %s] = %sprepareCompressArgs11(%s, %s, %d, %s);\n', ...
        NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, VarCondition, axis, NumDimX),...
        sprintf('%s = %s(Indices{:});\n', VarY, VarX),...
        ];
end
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "prepareCompressArgs11";
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

