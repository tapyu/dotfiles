function nodeTranslation = translateMod(~, nodeTranslation, node, IntegerTensorNames)
    %   Copyright 2024 The MathWorks, Inc.
    % Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
    % To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
    AttributeTable = cell2table({
        "fmod"       "INT"      true    0
        });
    % Parse the attributes
    [fmod] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
    C = node.output{1};
    A = node.input{1};
    B = node.input{2};
    [VarA, NumDimA]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, A);
    [VarB, ~]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, B);
    [VarC, NumDimC]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, C);

    if ismember(A, IntegerTensorNames) || ismember(B, IntegerTensorNames)
        nodeTranslation.IntegerOutputTensorNames = string(C);
    end

    command = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('[%s, %s] = %sonnxMod(%s, %s, %d, %s);\n', ...
        VarC, NumDimC, getPackageName(nodeTranslation.GraphTranslation), VarA, VarB, fmod, NumDimA)];
    nodeTranslation.MCode                   = string(command);
    nodeTranslation.IncludedFunctionNames   = "onnxMod";
end