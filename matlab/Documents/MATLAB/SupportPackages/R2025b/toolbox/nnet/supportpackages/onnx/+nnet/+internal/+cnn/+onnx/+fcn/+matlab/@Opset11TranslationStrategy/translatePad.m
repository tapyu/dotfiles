function nodeTranslation = translatePad(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2026 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "mode"      "STRING"    true    "constant"
    });
% Parse the attributes
[mode] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% Y = onnxPad(X, padding, value)
Y     	= node.output{1};
X    	= node.input{1};
pads	= node.input{2};
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarPads, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, pads);

if numel(node.input) > 2 && ~isempty(node.input{3})
    constant_valueStr = node.input{3};
    [VarConstant_valueStr,~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, constant_valueStr);
else
    VarConstant_valueStr = '0';
end
axes = sprintf("dlarray([0:%s]')",NumDimX);
fcn = "onnxPad";
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s] = %sonnxPad(%s, %s, %s, ''%s'', %s, %s);\n', ...
    VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, VarPads, VarConstant_valueStr, mode, axes, NumDimX)];
nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = fcn;
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end