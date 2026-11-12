function nodeTranslation = translateSlice(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2024 The MathWorks, Inc.

Y       = node.output{1};
X       = node.input{1};
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);

% Y = slice(X), with attributes: Axes, End, Starts. (Steps are always 1).
% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axes"     "INTS"      true     []
    "ends"     "INTS"      false    []
    "starts"   "INTS"      false    []
    });
% Parse the attributes
[Axes, Ends, Starts] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
Steps = '';

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[Indices, %s] = %sprepareSliceArgs(%s, [%s], [%s], [%s], [%s], %s);\n',...
    NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, ...
    strjoin(string(Starts), ', '), strjoin(string(Ends), ', '), strjoin(string(Axes), ', '), Steps, NumDimX),...
    sprintf('%s = %s(Indices{:});\n', VarY, VarX),...
    ];

nodeTranslation.IncludedFunctionNames   = "prepareSliceArgs";
nodeTranslation.MCode                   = string(command);
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end