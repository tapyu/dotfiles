function nodeTranslation = translateHardSwish(~, nodeTranslation, node, ~)
    % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
    % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
    %   Copyright 2024 The MathWorks, Inc.

    Y = node.output{1};
    X = node.input{1};
    [varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
    [varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
    command = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('[%s, %s] = %sonnxHardSigmoid(%s, %f, %f, %s);\n', varY, numDimsY, ...
        getPackageName(nodeTranslation.GraphTranslation), varX, 1/6, 0.5, numDimsX),...
        sprintf('%s = %s .* %s;\n', varY, varX, varY);
        ];
    nodeTranslation.MCode                   = string(command);
    nodeTranslation.IncludedFunctionNames   = "onnxHardSigmoid";
end