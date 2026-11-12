function nodeTranslation = translateResize(~, nodeTranslation, node, IntegerTensorNames)                    % Introduced in opset 10
%   Copyright 2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "mode"  "STRING"    true    "nearest"
    });
% Parse the attributes
[mode] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
if ~ismember(mode, ["nearest", "linear"])
    nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationError(node, message("nnet_cnn_onnx:onnx:UnsupportedAttributeValue", mode, 'mode'));
    return;
end
% Gen code
Y = node.output{1};
X = node.input{1};
scales  = node.input{2};

DLTScales = nnet.internal.cnn.onnx.fcn.uniqueCoderName('DLTScales');
dataFormat = nnet.internal.cnn.onnx.fcn.uniqueCoderName('dataFormat');
Method = nnet.internal.cnn.onnx.fcn.uniqueCoderName('Method');

varXTempName        = nnet.internal.cnn.onnx.fcn.uniqueCoderName('X');
varYTempName        = nnet.internal.cnn.onnx.fcn.uniqueCoderName('Y');

[VarX, NumDimX]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumDimY]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarScales, ~]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, scales);

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s, %s, %s] = %sprepareResize10Args(%s, "%s", %s);\n', ...
    DLTScales, dataFormat, Method,...
    NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarScales, mode, NumDimX),...  
    sprintf('%s = dlarray(single(%sextractIfDlarray(%s)));\n', varXTempName, getPackageName(nodeTranslation.GraphTranslation), VarX),...
    sprintf('%s = dlresize(%s, ''Scale'', %s(1:end-1), ''DataFormat'', %s, ''Method'', %s, ''GeometricTransformMode'', ''asymmetric'', ''NearestRoundingMode'', ''onnx-10'');\n', ...
    varYTempName, varXTempName, DLTScales, dataFormat, Method),...
    sprintf('%s = %sextractIfDlarray(%s);\n', VarY, getPackageName(nodeTranslation.GraphTranslation), varYTempName),...
    ];
nodeTranslation.IncludedFunctionNames = "prepareResize10Args";
nodeTranslation.MCode = string(command);
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end