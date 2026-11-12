function nodeTranslation = translateConvTranspose(~, nodeTranslation, node, ~)
%   Copyright 2023-2026 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "auto_pad"      "STRING"    true    "NOTSET"
    "dilations"     "INTS"      true    []
    "group"         "INT"       true    1
    "kernel_shape"  "INTS"      true    []
    "output_padding" "INTS"     true    []
    "output_shape"  "INTS"      true    []
    "pads"          "INTS"      true    []
    "strides"       "INTS"      true    []
    });
% Parse the attributes
[auto_pad, dilations, group, ~, output_padding, output_shape, pads, strides] = ...
    nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% auto_pad, pads
if auto_pad ~= "NOTSET"
    switch auto_pad
        case "SAME_UPPER"
            pads = "same";
        case "SAME_LOWER"
            pads = "same";
            nodeTranslation.Issues = [nodeTranslation.Issues nnet.internal.cnn.onnx.NodeTranslationWarning(node,...
                message("nnet_cnn_onnx:onnx:AutoPadSameLower", node.name))];
        case "VALID"
            % Leave default
        otherwise
            % Leave default
    end
elseif ~isempty(pads)
    pads = double(pads);
end
% group
numWtGroups = group;
% output_padding
if ~isempty(output_padding) && ~all(output_padding == 0) % output_padding of all zeros should not have any effect
    nodeTranslation.Issues = [nodeTranslation.Issues nnet.internal.cnn.onnx.NodeTranslationError(node,...
        message('nnet_cnn_onnx:onnx:UnsupportedAttribute', 'output_padding'))];
    return;
end
% output_shape
if ~isempty(output_shape)
    output_shape = double(output_shape);
end
% Y = dltranspconv(X,WEIGHTS,BIAS,...)
Y            = node.output{1};
X            = node.input{1};
WEIGHTS      = node.input{2};
setInitializerType(nodeTranslation.GraphTranslation,'Learnables',WEIGHTS);
setInitializerType(nodeTranslation.GraphTranslation,'Learnables',X);
[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[varWeights, numDimsWeights] = getVarAndNumDimName(nodeTranslation.GraphTranslation, WEIGHTS);
if numel(node.input) > 2 && ~isempty(node.input{3})
    BiasTerm = node.input{3};
    setInitializerType(nodeTranslation.GraphTranslation,'Learnables',BiasTerm);
    [varBias, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, BiasTerm);
else
    varBias = "''";
end

StrideName   = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Stride']);
DilationName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'DilationFactor']);
PadsName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Pads']);
OutputShapeName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'OutputShape']);
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', StrideName, nnet.internal.cnn.onnx.fcn.RankedArray(strides,1));
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', DilationName, nnet.internal.cnn.onnx.fcn.RankedArray(dilations,1));
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', PadsName, nnet.internal.cnn.onnx.fcn.RankedArray(pads,1));
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', OutputShapeName, nnet.internal.cnn.onnx.fcn.RankedArray(output_shape,1));

[varStride, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, StrideName);
[varDilation, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, DilationName);
[varPads, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, PadsName);
[varOutputShape, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, OutputShapeName);

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[weights, bias, stride, dilation, cropping, dataFormat, %s] = %sprepareConvTransposeArgs(%s, %s, %s, %s, %s, "%s", %s, %d, size(%s), %s, %s);\n',...
    numDimsY, getPackageName(nodeTranslation.GraphTranslation), varWeights, varBias, varStride, varDilation, varPads, string(auto_pad), varOutputShape, numWtGroups, varX, numDimsX, numDimsWeights),...
    sprintf('%s = dltranspconv(%s, weights, bias, ''Stride'', stride, ''DilationFactor'', dilation, ''Cropping'', cropping, ''DataFormat'', dataFormat);\n',...
    varY, varX),...
    ];

nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "prepareConvTransposeArgs";
end

