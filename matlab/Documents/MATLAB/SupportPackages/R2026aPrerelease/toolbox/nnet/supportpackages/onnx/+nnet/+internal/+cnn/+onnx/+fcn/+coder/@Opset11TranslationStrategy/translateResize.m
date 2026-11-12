function nodeTranslation = translateResize(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2024-2026 The MathWorks, Inc.

% Resize-11 adds 5 attributes and 2 inputs.
% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "coordinate_transformation_mode"    "STRING"    true    "half_pixel"
    "cubic_coeff_a"                     "FLOAT"     true    -0.75
    "exclude_outside"                   "INT"       true    0
    "extrapolation_value"               "FLOAT"     true    0.0
    "mode"                              "STRING"    true    "nearest"
    "nearest_mode"                      "STRING"    true    "round_prefer_floor"
    });
% Parse the attributes
[coordinate_transformation_mode,cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode] ...
    = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);

if isequal(coordinate_transformation_mode, "pytorch_half_pixel")
    coordinate_transformation_mode = "half_pixel";
end

if ~ismember(coordinate_transformation_mode, ["half_pixel", "asymmetric"]) || ...
        ~isequal(cubic_coeff_a, -0.75) || ...
        ~isequal(exclude_outside, 0) || ...
        ~isequal(extrapolation_value, 0.0) || ...
        ~ismember(nearest_mode, ["floor", "ceil", "round_prefer_floor", "round_prefer_ceil"])
    nodeTranslation.Issues = [nodeTranslation.Issues nnet.internal.cnn.onnx.NodeTranslationError(node,...
        message("nnet_cnn_onnx:onnx:UnsupportedAttributeValue", mode, 'mode', node.op_type))];
    return;
end
% if the nearest_mode is "floor", "ceil", "round_prefer_floor" and thrown out a warning
if ismember(nearest_mode, ["floor", "ceil", "round_prefer_floor"])
    nodeTranslation.Issues = [nodeTranslation.Issues nnet.internal.cnn.onnx.NodeTranslationWarning(node,...
        message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyImport", nearest_mode, 'nearest_mode', 'onnx-10'))];
end
% if the mode is "cubic", set the mode to be "linear", and throw out a warning
if isequal(mode, "cubic")
    mode = "linear";
    nodeTranslation.Issues = [nodeTranslation.Issues nnet.internal.cnn.onnx.NodeTranslationWarning(node,...
        message("nnet_cnn_onnx:onnx:UnsupportedResizePropertyImport", 'cubic', 'mode', 'linear'))];
end

% Gen code
Y = node.output{1};
X = node.input{1};
roi = node.input{2};

DLTScales = nnet.internal.cnn.onnx.fcn.uniqueCoderName('DLTScales');
DLTSizes = nnet.internal.cnn.onnx.fcn.uniqueCoderName('DLTSizes');
dataFormat = nnet.internal.cnn.onnx.fcn.uniqueCoderName('dataFormat');
Method = nnet.internal.cnn.onnx.fcn.uniqueCoderName('Method');
GeometricTransformMode = nnet.internal.cnn.onnx.fcn.uniqueCoderName('GeometricTransformMode');
NearestRoundingMode = nnet.internal.cnn.onnx.fcn.uniqueCoderName('NearestRoundingMode');

varXTempName        = nnet.internal.cnn.onnx.fcn.uniqueCoderName('X');
varYTempName        = nnet.internal.cnn.onnx.fcn.uniqueCoderName('Y');

[VarX, NumdimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumdimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
if isempty(roi)
    VarRoi = "[]";
else
	[VarRoi, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, roi); 
end
% scales can be the empty string
if isempty(node.input{3})
    VarScales = "[]";
else
    scalesStr  = node.input{3};
    if isInitializer(nodeTranslation.GraphTranslation, scalesStr)
        [~, rankedArray] = getInitializerValue(nodeTranslation.GraphTranslation, scalesStr);
        setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', scalesStr, rankedArray);
    end
    [VarScales, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, scalesStr);
end
% sizes can be the empty string or missing
if numel(node.input)<4 || isempty(node.input{4})
    VarSizes = "[]";
else
    sizesStr  = node.input{4};
    if isInitializer(nodeTranslation.GraphTranslation, sizesStr)
        [~, rankedArray] = getInitializerValue(nodeTranslation.GraphTranslation, sizesStr);
        setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', sizesStr, rankedArray);
    end
    [VarSizes, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, sizesStr);
end

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s, %s, %s, %s, %s, %s] = %sprepareResize11Args(%s, %s, %s, "%s", "%s", "%s", %s);\n', ...
    DLTScales, DLTSizes, dataFormat, Method, GeometricTransformMode, NearestRoundingMode,...
    NumdimY, getPackageName(nodeTranslation.GraphTranslation), VarRoi, VarScales, VarSizes, coordinate_transformation_mode, mode, nearest_mode, NumdimX),...
    sprintf('%s = dlarray(single(%sextractIfDlarray(%s)));\n', varXTempName, getPackageName(nodeTranslation.GraphTranslation), VarX),...    
    sprintf('if isempty(%s)\n', DLTScales),...    
    sprintf('%s = dlresize(%s, ''OutputSize'', %s(1:end-1), ''DataFormat'', %s, ''Method'', %s, ''GeometricTransformMode'', %s, ''NearestRoundingMode'', %s);\n', ...
    varYTempName, varXTempName, DLTSizes, dataFormat, Method, GeometricTransformMode, NearestRoundingMode),...
    sprintf('else\n'),...
    sprintf('%s = dlresize(%s, ''Scale'', %s(1:end-1), ''DataFormat'', %s, ''Method'', %s, ''GeometricTransformMode'', %s, ''NearestRoundingMode'', %s);\n', ...
    varYTempName, varXTempName, DLTScales, dataFormat, Method, GeometricTransformMode, NearestRoundingMode),...
    sprintf('end\n'),...
    sprintf('%s = %sextractIfDlarray(%s);\n', VarY, getPackageName(nodeTranslation.GraphTranslation), varYTempName),...
    ];
nodeTranslation.IncludedFunctionNames = "prepareResize11Args";
nodeTranslation.MCode = string(command);
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end