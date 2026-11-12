function nodeTranslation = translateGatherND(~, nodeTranslation, node, IntegerTensorNames)
    %   Copyright 2024-2026 The MathWorks, Inc.

    % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
    % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
    AttributeTable = cell2table({
        "batch_dims"  "INT"   true    0
        });
    
    % Parse the attributes
    batch_dims = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
    Y      = node.output{1};
    X      = node.input{1};
    indices = node.input{2};
    [VarX, NumdimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
    [VarY, NumdimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
    [VarIndices, NumdimIndices] = getVarAndNumDimName(nodeTranslation.GraphTranslation, indices);

    command = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('[%s, %s] = %sonnxGatherND(%s, %s, %d, %s, %s);\n', ...
        VarY, NumdimY, getPackageName(nodeTranslation.GraphTranslation), VarX, VarIndices, batch_dims, NumdimX, NumdimIndices),...
        ];
    nodeTranslation.MCode = string(command);
    nodeTranslation.IncludedFunctionNames = "onnxGatherND";
    if ismember(X, IntegerTensorNames)
        nodeTranslation.IntegerOutputTensorNames = string(Y);
    end
end