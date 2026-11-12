function nodeTranslation = translateHardSigmoid(~, nodeTranslation, node, ~)
    % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
    % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
    %   Copyright 2024 The MathWorks, Inc.

    AttributeTable = cell2table({
        "alpha"         "FLOAT"       true    0.2
        "beta"          "FLOAT"       true    0.5
        });
    % Parse the attributes
    [alpha, beta] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
    Y = node.output{1};
    X = node.input{1};
    [varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
    [varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
    command = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('[%s, %s] = %sonnxHardSigmoid(%s, %f, %f, %s);\n', varY, numDimsY, ...
        getPackageName(nodeTranslation.GraphTranslation), varX, alpha, beta, numDimsX)
        ];
    nodeTranslation.MCode                   = string(command);
    nodeTranslation.IncludedFunctionNames   = "onnxHardSigmoid";
end