function nodeTranslation = translateScatterND(~, nodeTranslation, node, IntegerTensorNames)       % Updated in opset 16
    %   Copyright 2024 The MathWorks, Inc.

    % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
    % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
    AttributeTable = cell2table({
        "reduction" "STRING"    true    "none"
        });
    % Parse the attributes
    [reduction] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
    Y       = node.output{1};
    X       = node.input{1};
    Indices	= node.input{2};
    Updates	= node.input{3};
    [VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
    [VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
    [VarIndices, NumDimIndices] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Indices);
    [VarUpdates, NumDimUpdates] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Updates);
    command = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('[%s, %s] = %sonnxScatterND(%s, %s, %s, "%s", %s, %s, %s);\n', ...
        VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, VarIndices, VarUpdates, reduction, NumDimX, NumDimIndices, NumDimUpdates),...
        ];
    nodeTranslation.MCode                   = string(command);
    nodeTranslation.IncludedFunctionNames   = "onnxScatterND";
    if ismember(X, IntegerTensorNames)
        nodeTranslation.IntegerOutputTensorNames = string(Y);
    end
end