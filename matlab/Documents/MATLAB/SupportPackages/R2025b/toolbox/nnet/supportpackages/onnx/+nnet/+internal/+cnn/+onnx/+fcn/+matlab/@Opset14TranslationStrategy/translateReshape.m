function nodeTranslation = translateReshape(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2026 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "allowzero"	"INT"    true    0
    });
% Parse the attributes
allowzero = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
B     = node.output{1};
A     = node.input{1};
SHAPE = node.input{2};
[VarB, NumDimB] = getVarAndNumDimName(nodeTranslation.GraphTranslation, B);
[VarA, NumDimA] = getVarAndNumDimName(nodeTranslation.GraphTranslation, A);
[VarShape, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, SHAPE);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[shape, %s] = %sprepareReshapeArgs(%s, %s, %s, %d);\n', NumDimB,...
    getPackageName(nodeTranslation.GraphTranslation), VarA, VarShape, NumDimA, allowzero),...
    sprintf('%s = reshape(%s, shape{:});\n', VarB, VarA),...
    ];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "prepareReshapeArgs";
if ismember(A, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(B);
end

end
