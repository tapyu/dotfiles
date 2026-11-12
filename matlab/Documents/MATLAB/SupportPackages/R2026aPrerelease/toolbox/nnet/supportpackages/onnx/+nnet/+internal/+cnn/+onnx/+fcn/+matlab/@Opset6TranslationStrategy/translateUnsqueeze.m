function nodeTranslation = translateUnsqueeze(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axes"     "INTS"      false    []
    });
% Parse the attributes
[axes] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% Y = unsqueeze(X)
Y       = node.output{1};
X       = node.input{1};
AxesName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Axes']);
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', AxesName, nnet.internal.cnn.onnx.fcn.RankedArray(axes, 2));

[VarAxes, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, AxesName);
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[shape, %s] = %sprepareUnsqueezeArgs(%s, %s, %s);\n', ...
    NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, VarAxes, NumDimX),...
    sprintf('%s = reshape(%s, shape);\n', VarY, VarX),...
    ];
nodeTranslation.IncludedFunctionNames = "prepareUnsqueezeArgs";
nodeTranslation.MCode = string(command);
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end