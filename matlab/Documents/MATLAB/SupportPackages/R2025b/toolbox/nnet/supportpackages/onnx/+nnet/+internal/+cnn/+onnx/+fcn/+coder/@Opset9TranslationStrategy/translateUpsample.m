function nodeTranslation = translateUpsample(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024 The MathWorks, Inc.

% OPSET 9 version: 'scales' is now an input instead of an attribute.
% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "mode"      "STRING"    true    "nearest"
    });
% Parse the attributes
[mode] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
if ~isequal(mode, "nearest")
    nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationWarning(node,...
        message("nnet_cnn_onnx:onnx:ModeNearest"));
    mode = "nearest";
end
% Y = onnxUpsample(X, scales, mode)
Y         	= node.output{1};
X         	= node.input{1};
scales     	= node.input{2};
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarX, NumDimX]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarScales, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, scales);

DLTScales = nnet.internal.cnn.onnx.fcn.uniqueCoderName('DLTScales');
DLTSizes = nnet.internal.cnn.onnx.fcn.uniqueCoderName('DLTSizes');
dataFormat = nnet.internal.cnn.onnx.fcn.uniqueCoderName('dataFormat');
Method = nnet.internal.cnn.onnx.fcn.uniqueCoderName('Method');
GeometricTransformMode = nnet.internal.cnn.onnx.fcn.uniqueCoderName('GeometricTransformMode');
NearestRoundingMode = nnet.internal.cnn.onnx.fcn.uniqueCoderName('NearestRoundingMode');

varXTempName        = nnet.internal.cnn.onnx.fcn.uniqueCoderName('X');
varYTempName        = nnet.internal.cnn.onnx.fcn.uniqueCoderName('Y');

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s, %s, %s, %s, %s, %s] = %sprepareResize11Args([], %s, dlarray([]), "half_pixel", "%s", "round", %s);\n', ...
    DLTScales, DLTSizes, dataFormat, Method, GeometricTransformMode, NearestRoundingMode,...
    NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarScales, mode, NumDimX),...
    printf('%s = dlarray(single(%sextractIfDlarray(%s)));\n', varXTempName, getPackageName(nodeTranslation.GraphTranslation), VarX),...        
    sprintf('%s = dlresize(%s, ''Scale'',  %s(1:end-1), ''DataFormat'', %s, ''Method'', %s, ''GeometricTransformMode'', %s, ''NearestRoundingMode'', %s);\n', ...
    varYTempName, varXTempName, DLTScales, dataFormat, Method, GeometricTransformMode, NearestRoundingMode),...
    sprintf('%s = %sextractIfDlarray(%s);\n', VarY, getPackageName(nodeTranslation.GraphTranslation), varYTempName),...
    ];
nodeTranslation.MCode = string(command);
nodeTranslation.IncludedFunctionNames = "prepareResize11Args";
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

