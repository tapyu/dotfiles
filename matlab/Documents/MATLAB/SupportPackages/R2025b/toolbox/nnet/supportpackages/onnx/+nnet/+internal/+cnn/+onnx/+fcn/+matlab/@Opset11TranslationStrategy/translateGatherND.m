function nodeTranslation = translateGatherND(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024-2026 The MathWorks, Inc.
    Y      = node.output{1};
    X      = node.input{1};
    indices = node.input{2};
    [VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
    [VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
    [VarIndices, NumDimIndices] = getVarAndNumDimName(nodeTranslation.GraphTranslation, indices);
    command = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('[%s, %s] = %sonnxGatherND(%s, %s, 0, %s, %s);\n', ... 
        VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, VarIndices, NumDimX, NumDimIndices),...
        ];
    nodeTranslation.MCode = string(command);
    nodeTranslation.IncludedFunctionNames = "onnxGatherND";
    if ismember(X, IntegerTensorNames)
        nodeTranslation.IntegerOutputTensorNames = string(Y);
    end
end