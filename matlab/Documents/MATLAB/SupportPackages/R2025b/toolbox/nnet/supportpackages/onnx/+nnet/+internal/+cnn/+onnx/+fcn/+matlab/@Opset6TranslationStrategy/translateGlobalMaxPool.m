function nodeTranslation = translateGlobalMaxPool(~, nodeTranslation, node, ~)
%   Copyright 2023-2024 The MathWorks, Inc.

Y        = node.output{1};
X        = node.input{1};
[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[poolsize, dataFormat, %s] = %sprepareGlobalMaxPoolArgs(%s, %s);\n',...
    numDimsY, getPackageName(nodeTranslation.GraphTranslation), varX, numDimsX),...
    sprintf('%s = maxpool(%s, poolsize, ''DataFormat'', dataFormat);\n', varY, varX),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "prepareGlobalMaxPoolArgs";
end

