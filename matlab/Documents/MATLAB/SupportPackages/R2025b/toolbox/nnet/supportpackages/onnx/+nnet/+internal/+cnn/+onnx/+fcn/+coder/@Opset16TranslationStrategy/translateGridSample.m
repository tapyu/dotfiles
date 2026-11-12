function nodeTranslation = translateGridSample(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024-2026 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "align_corners"      "INT"       true    0
    "mode"               "STRING"    true    "bilinear"
    "padding_mode"       "STRING"    true    "zeros"
    });
% Parse the attributes
[align_corners, mode, padding_mode] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
Y       = node.output{1};
X       = node.input{1};
grid	= node.input{2};
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarGrid, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, grid);
% numDim of X must be 4
if padding_mode ~="zeros" || mode ~= "bilinear" || ~ismember(align_corners, [0, 1])
    command = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('nnet.internal.cnn.coder.onnx.codererror(true, ''nnet_cnn_onnx:onnx:UnsupportedOperatorForCodegen'',''%s'');\n', node.op_type),...
        ];
else
    command = [
        sprintf('%% %s:\n', node.op_type),...
        sprintf('[%s, %s] = %sonnxGridSample(%s, %s, %s, %d, "%s", "%s");\n', ...
        VarY, NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarX, NumDimX, VarGrid, align_corners, mode, padding_mode),...
        ];
    nodeTranslation.IncludedFunctionNames   = "onnxGridSample";
    if ismember(X, IntegerTensorNames)
        nodeTranslation.IntegerOutputTensorNames = string(Y);
    end
end
nodeTranslation.MCode                   = string(command);
end