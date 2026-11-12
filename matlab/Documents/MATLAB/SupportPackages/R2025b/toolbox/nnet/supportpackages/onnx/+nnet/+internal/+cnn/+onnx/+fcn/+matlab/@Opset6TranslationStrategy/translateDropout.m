function nodeTranslation = translateDropout(~, nodeTranslation, node, ~)
%   Copyright 2023-2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "ratio"      "FLOAT"    true    0.5
    "is_test"    "INT"      true    0
    });
% Parse the attributes
[ratio, ~] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
X = node.input{1};
Y = node.output{1};
[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);

switch numel(node.output)
    case 1
        command = [
            sprintf('%% %s:\n', node.op_type),...
            sprintf('[%s, %s] = %sonnxDropout7(%s, %f, Training, %s);\n', ...
            varY, numDimsY, getPackageName(nodeTranslation.GraphTranslation), varX, ratio, numDimsX),...
            ];
    case 2
        Mask = node.output{2};
        [varMask, numDimsMask] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Mask);
        command = [
            sprintf('%% %s:\n', node.op_type),...
            sprintf('[%s, %s, %s, %s] = %sonnxDropout7(%s, %f, Training, %s);\n',...
            varY, numDimsY, varMask, numDimsMask, getPackageName(nodeTranslation.GraphTranslation), varX, ratio, numDimsX),...
            ];
end
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "onnxDropout7";
end

