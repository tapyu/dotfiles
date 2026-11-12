function nodeTranslation = translateLogSoftmax(~, nodeTranslation, node, ~)
%   Copyright 2025 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axis"         "INT"       true    1
    });
% Parse the attributes
[axis] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
Y     = node.output{1};
X     = node.input{1};
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[dim1, dim2, origSize, %s] = %sprepareSoftmaxArgs(%s, %d, %s);\n', ...
    NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, axis, NumDimX),...
    sprintf('%s = reshape(%s, dim1, dim2);\n', VarY, VarX),...
    sprintf('%s = log(softmax(%s, ''DataFormat'', ''CB''));\n', VarY, VarY),...
    sprintf('%s = reshape(%s, origSize);\n', VarY, VarY),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "prepareSoftmaxArgs";
end