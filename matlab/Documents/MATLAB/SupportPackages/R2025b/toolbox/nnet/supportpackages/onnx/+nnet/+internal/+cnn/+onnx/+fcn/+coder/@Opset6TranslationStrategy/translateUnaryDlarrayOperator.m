function nodeTranslation = translateUnaryDlarrayOperator(~, nodeTranslation, node, MATLABFcn, IntegerTensorNames)
%   Copyright 2024 The MathWorks, Inc.

Y = node.output{1};
X = node.input{1};
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarX, NumDimX]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
XTempName        = nnet.internal.cnn.onnx.fcn.uniqueCoderName('X');
YTempName        = nnet.internal.cnn.onnx.fcn.uniqueCoderName('Y');
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('%s = dlarray(%sextractIfDlarray(%s));\n', XTempName, getPackageName(nodeTranslation.GraphTranslation), VarX),...
    sprintf('%s = %s(%s);\n', YTempName, MATLABFcn, XTempName),...
    sprintf('%s = %sextractIfDlarray(%s);\n', VarY, getPackageName(nodeTranslation.GraphTranslation), YTempName),...
    sprintf('%s = %s;\n', NumDimY, NumDimX),...
    ];
nodeTranslation.MCode = string(command);
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

