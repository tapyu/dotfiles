function nodeTranslation = translatePad(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "mode"      "STRING"    true    "constant"
    "pads"  	"INTS"      false 	[]
    "value" 	"FLOAT"     true    0
    });
% Parse the attributes
[mode, pads, value] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% Y = onnxPad(X, padding, value)
Y         	= node.output{1};
X         	= node.input{1};
PaddingName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Padding']);
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', PaddingName, nnet.internal.cnn.onnx.fcn.RankedArray(pads,2));

[VarX, NumDimX]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumDimY]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarPadding, ~]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, PaddingName);
axes = sprintf("dlarray([0:%s]')",NumDimX);
fcn = "onnxPad";
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sonnxPad(%s, %s, %f, ''%s'', %s, %s);\n', ...
    VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, VarPadding, value, mode, axes, NumDimX)];

nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = fcn;
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end