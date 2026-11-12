function nodeTranslation = translateGridSample(~, nodeTranslation, node, ~)
    %   Copyright 2024-2025 The MathWorks, Inc.

    % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
    % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
    AttributeTable = cell2table({
        "align_corners"      "INT"       true    0
        "mode"               "STRING"    true    "bilinear"
        "padding_mode"       "STRING"    true    "zeros"
        });
    % Parse the attributes
    [align_corners, mode, padding_mode] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
    Y       = node.output{1};
    X       = node.input{1};
    grid	= node.input{2};
    [VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
    [VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
    [VarGrid, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, grid);
    if ~ismember(padding_mode,["zeros","border","reflection"])
        nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationError(...
            node, message("nnet_cnn_onnx:onnx:UnknownAttributeSetting","padding_mode", node.op_type));
        return;
    end
    
    if ~ismember(mode,["bilinear","nearest","bicubic"])
        nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationError(...
            node, message("nnet_cnn_onnx:onnx:UnknownAttributeSetting","mode", node.op_type));
        return;
    end

    if ~ismember(align_corners, [0, 1])
        nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationError(...
            node, message("nnet_cnn_onnx:onnx:UnknownAttributeSetting","align_corners", node.op_type));
        return;
    end
    command = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('[%s, %s] = %sonnxGridSample(%s, %s, %s, %d, "%s", "%s");\n', ...
        VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, NumDimX, VarGrid, align_corners, mode, padding_mode),...
        ];
    nodeTranslation.MCode                   = string(command);
    nodeTranslation.IncludedFunctionNames   = "onnxGridSample";
end