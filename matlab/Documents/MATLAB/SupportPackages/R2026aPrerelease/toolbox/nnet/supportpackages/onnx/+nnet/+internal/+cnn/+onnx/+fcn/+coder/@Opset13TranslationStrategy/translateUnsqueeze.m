function nodeTranslation = translateUnsqueeze(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024-2026 The MathWorks, Inc.

% Y = unsqueeze(X, axes)
Y       = node.output{1};
X       = node.input{1};
axes    = node.input{2};
shape = nnet.internal.cnn.onnx.fcn.uniqueCoderName('shape');
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarAxes, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, axes);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sprepareUnsqueezeArgs(%s, %s, %s);\n', ...
    shape, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, VarAxes, NumDimX),...
    sprintf('%s = reshape(%s, %s);\n', VarY, VarX, shape),...
    ];
nodeTranslation.MCode = string(command);
nodeTranslation.IncludedFunctionNames = "prepareUnsqueezeArgs";
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end