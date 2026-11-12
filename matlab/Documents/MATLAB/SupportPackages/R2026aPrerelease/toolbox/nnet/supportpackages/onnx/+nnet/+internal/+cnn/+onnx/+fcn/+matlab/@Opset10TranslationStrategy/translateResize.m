function nodeTranslation = translateResize(~, nodeTranslation, node, IntegerTensorNames)                    % Introduced in opset 10
%   Copyright 2023-2026 The MathWorks, Inc.


% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "mode"  "STRING"    true    "nearest"
    });
% Parse the attributes
[mode] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
if ~ismember(mode, ["nearest", "linear"])
    nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationError(node, message("nnet_cnn_onnx:onnx:UnsupportedAttributeValue", mode, 'mode',node.op_type));
    return;
end
% Gen code
Y = node.output{1};
X = node.input{1};
scales  = node.input{2};

[VarX, NumDimX]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumDimY]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarScales, ~]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, scales);

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[DLTScales, dataFormat, Method, %s] = %sprepareResize10Args(%s, "%s", %s);\n', ...
    NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarScales, mode, NumDimX),...    
    sprintf('%s = dlresize(%s, ''Scale'', DLTScales, ''DataFormat'', dataFormat, ''Method'', Method, ''GeometricTransformMode'', ''asymmetric'', ''NearestRoundingMode'', ''onnx-10'');\n', ...
    VarY, VarX),...
    ];
nodeTranslation.IncludedFunctionNames = "prepareResize10Args";
nodeTranslation.MCode = string(command);
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end