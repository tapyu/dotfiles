function nodeTranslation = translateLRN(~, nodeTranslation, node, ~)
%   Copyright 2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "alpha"    "FLOAT"      true    .0001
    "beta"     "FLOAT"  	true    .75
    "bias"     "FLOAT"      true    1
    "size"     "INT"        false   []
    });
% Parse the attributes
[alpha, beta, bias, windowSize] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% Y = crosschannelnorm(X,CHANNELWINDOWSIZE,'DataFormat',FMT,...)
Y               = node.output{1};
X               = node.input{1};
dataFormat = nnet.internal.cnn.onnx.fcn.uniqueCoderName('dataFormat');
XTempName        = nnet.internal.cnn.onnx.fcn.uniqueCoderName('X');
YTempName        = nnet.internal.cnn.onnx.fcn.uniqueCoderName('Y');

[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('if %s > 2\n', numDimsX),...
    sprintf('%s = coder.const([repmat(''S'',[1 %s-2]),''CB'']);\n', dataFormat, numDimsX),...
    sprintf('%s = dlarray(%sextractIfDlarray(%s));\n', XTempName, getPackageName(nodeTranslation.GraphTranslation), varX),...
    sprintf('%s = nnet.internal.cnn.coder.layer.utils.normUtils.crosschannelnorm(dlarray(%s, %s), %d, %d, %d, %d);\n',...
    YTempName, XTempName, dataFormat, windowSize, alpha, beta, bias),...
    sprintf('%s = %sextractIfDlarray(%s);\n', varY, getPackageName(nodeTranslation.GraphTranslation), YTempName),...
    sprintf('else\n'),...
    sprintf('%s = %s;\n', varY, varX),...
    sprintf('end\n'),...
    sprintf('%s = %s;\n', numDimsY, numDimsX),...
    ];
nodeTranslation.MCode = string(command);
end

