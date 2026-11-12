function nodeTranslation = translateSqueeze(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axes"     "INTS"      true    []
    });
% Parse the attributes
[axes] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
AxesName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Axes']);
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', AxesName, nnet.internal.cnn.onnx.fcn.RankedArray(axes,2));

[VarAxes, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, AxesName);
% Y = Squeeze(X, axes)
Y       = node.output{1};
X       = node.input{1};
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
fcn = "onnxSqueeze";
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %s%s(%s, %s, %s);\n', VarY, NumDimY, ...
    getPackageName(nodeTranslation.GraphTranslation), fcn, VarX, VarAxes, NumDimX),...
    ];

nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = fcn;
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end