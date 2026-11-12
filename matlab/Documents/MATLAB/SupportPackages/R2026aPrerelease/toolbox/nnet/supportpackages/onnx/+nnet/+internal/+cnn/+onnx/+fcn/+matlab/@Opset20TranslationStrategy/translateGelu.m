function nodeTranslation = translateGelu(~, nodeTranslation, node, ~)
    %   Copyright 2024 The MathWorks, Inc.

    % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
    % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
    AttributeTable = cell2table({
        "approximate"      "STRING"       true    'none'
        });
    % Parse the attributes
    [approximate] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
    
    X = node.input{1};
    Y = node.output{1};
    [VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
    [VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
    code = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('%s = gelu(%s,"Approximation", "%s");\n', VarY, VarX, approximate),...
        sprintf('%s = %s;\n', NumDimY, NumDimX)
        ];
    
    nodeTranslation.MCode                   = string(code);
end