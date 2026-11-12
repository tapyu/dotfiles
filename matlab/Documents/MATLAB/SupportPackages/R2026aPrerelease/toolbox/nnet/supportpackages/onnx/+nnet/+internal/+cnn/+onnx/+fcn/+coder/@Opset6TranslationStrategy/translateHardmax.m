function nodeTranslation = translateHardmax(~, nodeTranslation, node, ~)
%   Copyright 2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axis"         "INT"       true    1
    });
% Parse the attributes
[axis] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% Y = hardmax(X)
Y     = node.output{1};
X     = node.input{1};
[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);

dim1Var = nnet.internal.cnn.onnx.fcn.uniqueCoderName('dim1');
dim2Var = nnet.internal.cnn.onnx.fcn.uniqueCoderName('dim2');
origSizeVar = nnet.internal.cnn.onnx.fcn.uniqueCoderName('origSize');
maxIndicesVar = nnet.internal.cnn.onnx.fcn.uniqueCoderName('maxInds');
linIndicesVar = nnet.internal.cnn.onnx.fcn.uniqueCoderName('linearIndices');

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s, %s, %s] = %sprepareHardmaxArgs(%s, %d, %s);\n',...
    dim1Var, dim2Var, origSizeVar, numDimsY, getPackageName(nodeTranslation.GraphTranslation), varX, axis, numDimsX),...
    sprintf('%s = reshape(%s, %s, %s);\n', varY, varX, dim1Var, dim2Var),...
    sprintf('[~,%s] = max(%s, [], 1);\n', maxIndicesVar, varY),...
    sprintf('%s = (0:%s-1)*%s + %s;\n', linIndicesVar, dim2Var, dim1Var, maxIndicesVar),...
    sprintf('%s(:) = 0;\n', varY),...
    sprintf('%s(%s) = 1;\n', varY, linIndicesVar),...
    sprintf('%s = reshape(%s, %s);\n', varY, varY, origSizeVar),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "prepareHardmaxArgs";
end

