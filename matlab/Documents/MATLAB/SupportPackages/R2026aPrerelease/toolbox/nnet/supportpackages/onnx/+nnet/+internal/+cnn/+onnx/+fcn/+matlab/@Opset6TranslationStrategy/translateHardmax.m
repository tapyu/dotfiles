function nodeTranslation = translateHardmax(~, nodeTranslation, node, ~)
%   Copyright 2023-2024 The MathWorks, Inc.

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
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[dim1, dim2, origSize, %s] = %sprepareHardmaxArgs(%s, %d, %s);\n',...
    numDimsY, getPackageName(nodeTranslation.GraphTranslation), varX, axis, numDimsX),...
    sprintf('%s = reshape(%s, dim1, dim2);\n', varY, varX),...
    sprintf('[~,I] = max(%s, [], 1);\n', varY),...
    sprintf('linearIndices = (0:dim2-1)*dim1 + I;\n'),...
    sprintf('%s(:) = 0;\n', varY),...
    sprintf('%s(linearIndices) = 1;\n', varY),...
    sprintf('%s = reshape(%s, origSize);\n', varY, varY),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "prepareHardmaxArgs";
end

