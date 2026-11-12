function nodeTranslation = translateShape(~, nodeTranslation, node, ~)
    %   Copyright 2024 The MathWorks, Inc.

    % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
    % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
    AttributeTable = cell2table({
        "start"	"INT"    true    0
        "end"   "INT"   true    []
        });
    % Parse the attributes
    [Start, End] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);   
    Y = node.output{1};
    X = node.input{1};
    [VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
    [VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
    if isempty(End)
        command = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('[%s, %s] = %sonnxShape(%s, %s, %d, %s+1);\n',...
        VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, NumDimX, Start, NumDimX),...
        ];
    else
        command = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('[%s, %s] = %sonnxShape(%s, %s, %d, %d);\n',...
        VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, NumDimX, Start, End),...
        ];
    end
    
    nodeTranslation.MCode                   = string(command);
    nodeTranslation.IncludedFunctionNames   = "onnxShape";
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end