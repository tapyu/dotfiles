function nodeTranslation = translateGather(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2026 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "axis"         "INT"       true    0
    });
% Parse the attributes
[axis] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
Y      = node.output{1};
X      = node.input{1};
idx    = node.input{2};
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VaridX, NumDimidX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, idx);
% Gather is a subsref then a reshape.
fcn = "onnxGather";
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %s%s(%s, %s, %d, %s, %s);\n', VarY, NumDimY, ...
    getPackageName(nodeTranslation.GraphTranslation), fcn, VarX, VaridX, axis, NumDimX, NumDimidX),...
    ];
nodeTranslation.IncludedFunctionNames = fcn;
nodeTranslation.MCode = string(command);
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end

end