function nodeTranslation = translateFlatten(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axis"         "INT"       true    1
    });
% Parse the attributes
[axis] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
Y     = node.output{1};
X     = node.input{1};
[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
dim1Name = nnet.internal.cnn.onnx.fcn.uniqueCoderName('dim1');
dim2Name = nnet.internal.cnn.onnx.fcn.uniqueCoderName('dim2');
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s, %s] = %sprepareFlattenArgs(%s, %d, %s);\n', ...
    dim1Name, dim2Name, numDimsY, getPackageName(nodeTranslation.GraphTranslation), varX, axis, numDimsX),...
    sprintf('%s = reshape(%s, %s, %s);\n', varY, varX, dim1Name, dim2Name),...
    ];
nodeTranslation.MCode = string(command);
nodeTranslation.IncludedFunctionNames = "prepareFlattenArgs";
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

