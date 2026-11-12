function nodeTranslation = translateGlobalAveragePool(~, nodeTranslation, node, ~)
%   Copyright 2023-2024 The MathWorks, Inc.

Y        = node.output{1};
X        = node.input{1};
[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[poolsize, dataFormat, %s] = %sprepareGlobalAveragePoolArgs(%s, %s);\n', ...
    numDimsY, getPackageName(nodeTranslation.GraphTranslation), varX, numDimsX),...    
    sprintf('%s = avgpool(%s, poolsize, ''DataFormat'', dataFormat);\n', varY, varX),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "prepareGlobalAveragePoolArgs";
end

