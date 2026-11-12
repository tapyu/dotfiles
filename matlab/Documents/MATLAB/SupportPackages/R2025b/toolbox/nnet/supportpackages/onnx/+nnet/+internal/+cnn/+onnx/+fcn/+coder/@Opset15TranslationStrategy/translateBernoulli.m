function nodeTranslation = translateBernoulli(~, nodeTranslation, node, ~)
    %   Copyright 2024 The MathWorks, Inc.

    % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
    % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
    AttributeTable = cell2table({
        "dtype"     "INT"      true    1  
        "seed"      "FLOAT"    true    []  
        });
    % Parse the attributes
    [dtype, seed] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
    X = node.input{1};
    Y = node.output{1};
    [VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
    [VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
    command = sprintf('%% %s:\n', node.op_type);
    if ~isempty(seed)
        command = [command, sprintf('rng(%f);\n', seed)];
    end
    command = [command,...
    sprintf('[%s, %s] = %sonnxBernoulli(%s, %s);\n',...
    VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, NumDimX),...
    ];
    % From TensorProto, dtype can be:
    %     UNDEFINED = 0;
    %     // Basic types.
    %     FLOAT = 1;   // float
    %     UINT8 = 2;   // uint8_t
    %     INT8 = 3;    // int8_t
    %     UINT16 = 4;  // uint16_t
    %     INT16 = 5;   // int16_t
    %     INT32 = 6;   // int32_t
    %     INT64 = 7;   // int64_t
    %     STRING = 8;  // string
    %     BOOL = 9;    // bool
    if ismember(dtype, [2 3 4 5 6 7])
        nodeTranslation.IntegerOutputTensorNames = string(Y);
    end
    nodeTranslation.MCode = string(command);
    nodeTranslation.IncludedFunctionNames   = "onnxBernoulli";
end