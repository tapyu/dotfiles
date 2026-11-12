function nodeTranslation = translateLogSoftmax(~, nodeTranslation, node, ~)
%   Copyright 2025 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axis"         "INT"       true    1
    });
% Parse the attributes
[axis] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% Y = LogSoftmax(X)
Y     = node.output{1};
X     = node.input{1};
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
dim1Name = nnet.internal.cnn.onnx.fcn.uniqueCoderName('dim1');
dim2Name = nnet.internal.cnn.onnx.fcn.uniqueCoderName('dim2');
origSizeName = nnet.internal.cnn.onnx.fcn.uniqueCoderName('origSize');
varY1TempName        = nnet.internal.cnn.onnx.fcn.uniqueCoderName('Y1');
varY2TempName        = nnet.internal.cnn.onnx.fcn.uniqueCoderName('Y2');

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s, %s, %s] = %sprepareSoftmaxArgs(%s, %d, %s);\n', ...
    dim1Name, dim2Name, origSizeName, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, axis, NumDimX),...
    sprintf('%s = reshape(%s, %s, %s);\n', varY1TempName, VarX, dim1Name, dim2Name),...
    sprintf('%s = log(softmax(dlarray(single(%s)), ''DataFormat'', ''CB''));\n', varY2TempName, varY1TempName),...
    sprintf('%s = reshape(%sextractIfDlarray(%s), %s);\n', VarY, getPackageName(nodeTranslation.GraphTranslation), varY2TempName, origSizeName),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "prepareSoftmaxArgs";
end
