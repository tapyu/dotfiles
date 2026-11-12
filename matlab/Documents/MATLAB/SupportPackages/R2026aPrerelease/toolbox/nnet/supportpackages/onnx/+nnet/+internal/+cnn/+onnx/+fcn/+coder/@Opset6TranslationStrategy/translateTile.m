function nodeTranslation = translateTile(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024 The MathWorks, Inc.

% B = Tile(A, repeats)
B     = node.output{1};
A     = node.input{1};
repeats = node.input{2};
[VarA, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, A);
[VarB, NumDimB] = getVarAndNumDimName(nodeTranslation.GraphTranslation, B);
[VarRepeats, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, repeats);

sizeTempName = nnet.internal.cnn.onnx.fcn.uniqueCoderName('sz');

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sprepareTileArgs(%s);\n', sizeTempName, NumDimB, ...
    getPackageName(nodeTranslation.GraphTranslation), VarRepeats),...
    sprintf('%s = repmat(%s, %s);\n', VarB, VarA, sizeTempName),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "prepareTileArgs";
if ismember(A, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(B);
end
end

