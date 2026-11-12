function nodeTranslation = translateAveragePool(~, nodeTranslation, node, ~)
%   Copyright 2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "auto_pad"          "STRING"    true    "NOTSET"
    "count_include_pad"	"INT"       true    0
    "kernel_shape"      "INTS"      false   []
    "pads"              "INTS"      true    []
    "strides"           "INTS"      true    []
    });
% Parse the attributes
[auto_pad, count_include_pad, kernel_shape, pads, strides] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
% auto_pad, pads
numSpatialDims = numel(kernel_shape);
padding = zeros(2*numSpatialDims,1);
if auto_pad ~= "NOTSET"
    switch auto_pad
        case "SAME_UPPER"
            padding = "same";
        case "SAME_LOWER"
            padding = "same";
            nodeTranslation.Issues = [nodeTranslation.Issues nnet.internal.cnn.onnx.NodeTranslationWarning(node,...
                message("nnet_cnn_onnx:onnx:AutoPadSameLower"))];
        case "VALID"
            % Leave default
        otherwise
            % Leave default
    end
elseif isempty(pads)
    padding = zeros(2*numSpatialDims,1);
else
    padding = double(pads);
end
% strides
if isempty(strides)
    strides = ones(numSpatialDims,1);
end
% Y = avgpool(X,POOLSIZE,...)
Y        = node.output{1};
X        = node.input{1};
PoolsizeName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'PoolSize']);
StrideName   = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Stride']);
PaddingName  = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Padding']);
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', PoolsizeName, nnet.internal.cnn.onnx.fcn.RankedArray(kernel_shape,1));
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', StrideName, nnet.internal.cnn.onnx.fcn.RankedArray(strides,1));
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', PaddingName, nnet.internal.cnn.onnx.fcn.RankedArray(padding,1));

[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[varPoolsize, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, PoolsizeName);
[varStride, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, StrideName);
[varPadding, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, PaddingName);

poolSizeTempName        = nnet.internal.cnn.onnx.fcn.uniqueCoderName('poolSize');
strideTempName          = nnet.internal.cnn.onnx.fcn.uniqueCoderName('stride');
paddingTempName         = nnet.internal.cnn.onnx.fcn.uniqueCoderName('padding');
paddingValueTempName    = nnet.internal.cnn.onnx.fcn.uniqueCoderName('paddingValue');
dataFormatTempName      = nnet.internal.cnn.onnx.fcn.uniqueCoderName('dataFormat');

varXTempName        = nnet.internal.cnn.onnx.fcn.uniqueCoderName('X');
varYTempName        = nnet.internal.cnn.onnx.fcn.uniqueCoderName('Y');

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s, %s, %s, %s, %s] = %sprepareAveragePoolArgs(%s, %s, %s, %d, %s);\n',...
    poolSizeTempName, strideTempName, paddingTempName, paddingValueTempName, dataFormatTempName,...
    numDimsY, getPackageName(nodeTranslation.GraphTranslation), varPoolsize, varStride, varPadding, count_include_pad, numDimsX),... 
    sprintf('%s = dlarray(single(%sextractIfDlarray(%s)));\n', varXTempName, getPackageName(nodeTranslation.GraphTranslation), varX),...
    sprintf('%s = avgpool(%s, %s, ''Stride'', %s, ''Padding'', %s, ''PaddingValue'', %s, ''DataFormat'', %s);\n', varYTempName, varXTempName,...
    poolSizeTempName, strideTempName, paddingTempName, paddingValueTempName, dataFormatTempName),...
    sprintf('%s = %sextractIfDlarray(%s);\n', varY, getPackageName(nodeTranslation.GraphTranslation), varYTempName),...
    ];

nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "prepareAveragePoolArgs";
end

