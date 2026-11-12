function nodeTranslation = translateGlobalAveragePool(~, nodeTranslation, node, ~)
%   Copyright 2024 The MathWorks, Inc.

Y        = node.output{1};
X        = node.input{1};
[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);

poolsizeTempName             = nnet.internal.cnn.onnx.fcn.uniqueCoderName('poolsize');
dataFormatTempName          = nnet.internal.cnn.onnx.fcn.uniqueCoderName('dataFormat');

varXTempName        = nnet.internal.cnn.onnx.fcn.uniqueCoderName('X');
varYTempName        = nnet.internal.cnn.onnx.fcn.uniqueCoderName('Y');

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s, %s] = %sprepareGlobalAveragePoolArgs(%s, %s);\n', ...
    poolsizeTempName, dataFormatTempName, numDimsY, getPackageName(nodeTranslation.GraphTranslation), varX, numDimsX),...  
    sprintf('%s = dlarray(single(%sextractIfDlarray(%s)));\n', varXTempName, getPackageName(nodeTranslation.GraphTranslation), varX),...    
    sprintf('%s = avgpool(%s, %s, ''DataFormat'', %s);\n', varYTempName, varXTempName, poolsizeTempName, dataFormatTempName),...
    sprintf('%s = %sextractIfDlarray(%s);\n', varY, getPackageName(nodeTranslation.GraphTranslation), varYTempName),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "prepareGlobalAveragePoolArgs";
end

