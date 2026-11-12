function nodeTranslation = translateSoftmax(~, nodeTranslation, node, ~)
%   Copyright 2023 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axis"         "INT"       true    -1
    });
% Parse the attributes
[axis] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% Y = softmax(X)
Y     = node.output{1};
X     = node.input{1};
[VarX, NumdimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumdimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sonnxSoftmax13(%s, %d, %s);\n', ...
    VarY, NumdimY, getPackageName(nodeTranslation.GraphTranslation), VarX, axis, NumdimX),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = ["onnxSoftmax13", "prepareReduceArgs"];

end

