function nodeTranslation = translateLayerNormalization(~, nodeTranslation, node, ~)
    %   Copyright 2024-2026 The MathWorks, Inc. 
    % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
    % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
    AttributeTable = cell2table({
        "axis"         "INT"        true    -1
        "epsilon"      "FLOAT"      true    1e-5
        "stash_type"   "INT"        true    1
        });
    % Parse the attributes
    [axis, epsilon, ~] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
    % Casting inputs to different datatype in the layer "stash_type" is not supported
    if numel(node.attribute) > 2
    nodeTranslation.Issues = [nodeTranslation.Issues nnet.internal.cnn.onnx.NodeTranslationWarning(node,...
                message('nnet_cnn_onnx:onnx:IgnoredAttributes',"stash_type",node.op_type))];
    end
    if epsilon <= 0
        epsilon = 1e-5;
        nodeTranslation.Issues = [nodeTranslation.Issues nnet.internal.cnn.onnx.NodeTranslationWarning(node,...
            message('nnet_cnn_onnx:onnx:BadEpsilon'))];
    end
    Epsilon         = double(epsilon);
    X               = node.input{1};
    ScaleName       = node.input{2};
    Y               = node.output{1};
    [VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
    [VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
    setInitializerType(nodeTranslation.GraphTranslation,'Learnables',ScaleName);
    [VarScale, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, ScaleName);
    if numel(node.input)>2
        OffsetName      = node.input{3};
        setInitializerType(nodeTranslation.GraphTranslation,'Learnables',OffsetName);
        [VarOffset, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, OffsetName);
        code = [sprintf('%% %s:\n', node.op_type),...
            sprintf('[%s, %s] = %sonnxLayerNormalization(%s, %s, %s, %i, %e, %s);\n',...
            VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, VarScale, VarOffset, axis, Epsilon, NumDimX)];
    else
        code = [sprintf('%% %s:\n', node.op_type),...
            sprintf('[%s, %s] = %sonnxLayerNormalization(%s, %s, [], %i, %e, %s);\n',...
            VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, VarScale, axis, Epsilon, NumDimX)];
    end

    nodeTranslation.MCode                   = string(code);
    nodeTranslation.IncludedFunctionNames   = "onnxLayerNormalization";
end