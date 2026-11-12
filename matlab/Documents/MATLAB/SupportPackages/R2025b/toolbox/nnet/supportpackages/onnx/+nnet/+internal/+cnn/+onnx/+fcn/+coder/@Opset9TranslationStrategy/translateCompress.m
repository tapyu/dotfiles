function nodeTranslation = translateCompress(~, nodeTranslation, node, IntegerTensorNames)    	% Introduced in opset 9
%   Copyright 2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axis"         "INT"       true    -1
    });
% Parse the attributes
[axis] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
Y       = node.output{1};
X       = node.input{1};
condition = node.input{2};
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarCondition, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, condition);
indicesVar = nnet.internal.cnn.onnx.fcn.uniqueCoderName('Indices');

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sprepareCompressArgs(%s, %s, %d, %s);\n', ...
    indicesVar, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, VarCondition, axis, NumDimX),...
    ];
if axis < 0
    command = [command, sprintf('%s = %s(%s);\n', VarY, VarX, indicesVar)];            % X is implicitly flattened first.
else
    command = [command, sprintf('%s = %s(%s{:});\n', VarY, VarX, indicesVar)];   % X is not flattened first.
end
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "prepareCompressArgs";
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

