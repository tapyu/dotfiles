function nodeTranslation = translateTranspose(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "perm"     "INTS"      true    []
    });
% Parse the attributes
[perm] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% B = permute(A, ORDER). We'll pass ORDER='' in the default case.
B        = node.output{1};
A        = node.input{1};
[VarA, NumDimA] = getVarAndNumDimName(nodeTranslation.GraphTranslation, A);
[VarB, NumDimB] = getVarAndNumDimName(nodeTranslation.GraphTranslation, B);
PermName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Perm']);
% Get the perm
if isempty(perm)
    VarOrder= "''";
else
    setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', PermName, nnet.internal.cnn.onnx.fcn.RankedArray(perm, 2));
    [VarOrder, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, PermName);
end
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[perm, %s] = %sprepareTransposeArgs(%s, %s);\n', NumDimB, getPackageName(nodeTranslation.GraphTranslation), VarOrder, NumDimA),...
    sprintf('if isempty(perm)\n'),...
    sprintf('%s = %s;\n', VarB, VarA),...
    sprintf('else\n'),...
    sprintf('%s = permute(%s, perm);\n', VarB, VarA),...
    sprintf('end\n'),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "prepareTransposeArgs";
if ismember(node.input{1}, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(B);
end
end

