function nodeTranslation = translateLRN(~, nodeTranslation, node, ~)
%   Copyright 2023-2024 The MathWorks, Inc.

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
[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('dataFormat = [repmat(''S'', 1, %s-2) ''CB''];\n', numDimsX),...    
    sprintf('%s = crosschannelnorm(%s, %d, ''Alpha'', %d, ''Beta'', %d, ''K'', %d, ''DataFormat'', dataFormat);\n',...
    varY, varX, windowSize, alpha, beta, bias),...
    sprintf('%s = %s;\n', numDimsY, numDimsX),...
    ];
nodeTranslation.MCode = string(command);
end

