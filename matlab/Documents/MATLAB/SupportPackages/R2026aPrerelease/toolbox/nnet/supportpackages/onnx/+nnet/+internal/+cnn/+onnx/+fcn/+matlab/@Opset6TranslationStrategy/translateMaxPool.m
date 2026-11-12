function nodeTranslation = translateMaxPool(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "auto_pad"          "STRING"    true    "NOTSET"
    "kernel_shape"      "INTS"      false   []
    "pads"              "INTS"      true    []
    "strides"           "INTS"      true    []
    });
% Parse the attributes
[auto_pad,  kernel_shape, pads, strides] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
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
% Y = maxpool(X,POOLSIZE,...)
Y        = node.output{1};
X        = node.input{1};
PoolsizeName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'PoolSize']);
StrideName   = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Stride']);
PaddingName  = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Padding']);
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', PoolsizeName, nnet.internal.cnn.onnx.fcn.RankedArray(kernel_shape,1));
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', StrideName, nnet.internal.cnn.onnx.fcn.RankedArray(strides,1));
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', PaddingName, nnet.internal.cnn.onnx.fcn.RankedArray(padding,1));

[VarX, NumDimX]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarY, NumDimY]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarPoolsize, ~]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, PoolsizeName);
[VarStride, ~]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, StrideName);
[VarPadding, ~]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, PaddingName);

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[poolsize, stride, padding, dataFormat, %s] = %sprepareMaxPool7Args(%s, %s, %s, %s);\n', ...
    NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarPoolsize, VarStride, VarPadding, NumDimX),...
    sprintf('%s = maxpool(%s, poolsize, ''Stride'', stride, ''Padding'', padding, ''DataFormat'', dataFormat);\n', VarY, VarX),...
    ];

nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "prepareMaxPool7Args";
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames = string(Y);
end
end

