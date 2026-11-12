function nodeTranslation = translateReshape(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024-2026 The MathWorks, Inc.

% B = reshape(A, SHAPE)
B     = node.output{1};
A     = node.input{1};
SHAPE = node.input{2};

[VarB, NumDimB] = getVarAndNumDimName(nodeTranslation.GraphTranslation, B);
[VarA, NumDimA] = getVarAndNumDimName(nodeTranslation.GraphTranslation, A);
[VarShape, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, SHAPE);
allowzero = 0; % Infer shape if it has value 0
shapeName = nnet.internal.cnn.onnx.fcn.uniqueCoderName('shape');
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sprepareReshapeArgs(%s, %s, %s, %d);\n', shapeName, NumDimB,...
    getPackageName(nodeTranslation.GraphTranslation), VarA, VarShape, NumDimA, allowzero),...
    sprintf('%s = reshape(%s, %s{:});\n', VarB, VarA, shapeName),...
    ];
nodeTranslation.IncludedFunctionNames   = "prepareReshapeArgs";
nodeTranslation.MCode                   = string(command);
if ismember(A, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(B);
end

end