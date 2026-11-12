function nodeTranslation = translateUpsample(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "mode"      "STRING"    true    "nearest"
    "scales"    "FLOATS"    false   []
    });
% Parse the attributes
[mode, scales] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
if ~isequal(mode, "nearest")
    nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationWarning(node,...
        message("nnet_cnn_onnx:onnx:ModeNearest"));
    mode = "nearest";
end
scalesName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Scales']);
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', scalesName, nnet.internal.cnn.onnx.fcn.RankedArray(scales,2));
[VarScales, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, scalesName);
% Y = onnxUpsample(X, scales, mode)
Y         	= node.output{1};
X         	= node.input{1};
[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[DLTScales, DLTSizes, dataFormat, Method, GeometricTransformMode, NearestRoundingMode, %s] = %sprepareResize11Args([], %s, dlarray([]), "half_pixel", "%s", "round", %s);\n', ...
    NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarScales, mode, NumDimX),...    
    sprintf('%s = dlresize(%s, ''Scale'', DLTScales, ''DataFormat'', dataFormat, ''Method'', Method, ''GeometricTransformMode'', GeometricTransformMode, ''NearestRoundingMode'', NearestRoundingMode);\n', ...
    VarY, VarX),...
    ];
nodeTranslation.MCode = string(command);
nodeTranslation.IncludedFunctionNames = "prepareResize11Args";
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

% Utilities:
