function nodeTranslation = translateTile(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2024 The MathWorks, Inc.

% B = Tile(A, repeats)
B     = node.output{1};
A     = node.input{1};
repeats = node.input{2};
[VarA, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, A);
[VarB, NumDimB] = getVarAndNumDimName(nodeTranslation.GraphTranslation, B);
[VarRepeats, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, repeats);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[sz, %s] = %sprepareTileArgs(%s);\n', NumDimB, ...
    getPackageName(nodeTranslation.GraphTranslation), VarRepeats),...
    sprintf('%s = repmat(%s, sz);\n', VarB, VarA),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "prepareTileArgs";
if ismember(A, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(B);
end
end

