function nodeTranslation = translateMaxPool(~, nodeTranslation, node, IntegerTensorNames)
%   Copyright 2023-2024 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "auto_pad"          "STRING"    true    "NOTSET"
    "kernel_shape"      "INTS"      false   []
    "pads"              "INTS"      true    []
    "storage_order"     "INT"       true    []          % new in opset 8
    "strides"           "INTS"      true    []
    });
% Parse the attributes
[auto_pad,  kernel_shape, pads, storage_order, strides] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
numSpatialDims = numel(kernel_shape);
padding = zeros(2*numSpatialDims,1);
% auto_pad, pads
if auto_pad ~= "NOTSET"
    switch auto_pad
        case "SAME_UPPER"
            padding = "same";
        case "SAME_LOWER"
            padding = "same";
            nodeTranslation.Issues = [nodeTranslation.Issues nnet.internal.cnn.onnx.NodeTranslationWarning(node,...
                message("nnet_cnn_onnx:onnx:AutoPadSameLower", node.name))];
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
% column-major not supported
if storage_order==1
    nodeTranslation.Issues = nnet.internal.cnn.onnx.NodeTranslationError(node,...
        message('nnet_cnn_onnx:onnx:AttributeMustHaveValue', 'MaxPool', 'storage_order', '0', node.name));
    return;
end
Y = node.output{1};
X = node.input{1};
PoolsizeName = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'PoolSize']);
StrideName   = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Stride']);
PaddingName  = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Padding']);
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', PoolsizeName, nnet.internal.cnn.onnx.fcn.RankedArray(kernel_shape,1));
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', StrideName, nnet.internal.cnn.onnx.fcn.RankedArray(strides,1));
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', PaddingName, nnet.internal.cnn.onnx.fcn.RankedArray(padding,1));

[VarY, NumDimY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
[VarX, NumDimX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[VarPoolsize, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, PoolsizeName);
[VarStride, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, StrideName);
[VarPadding, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, PaddingName);

switch numel(node.output)
    case 1
        % Single-output: Y = maxpool(X,POOLSIZE,...)
        command = [
            sprintf('%% %s:\n', node.op_type),...
            sprintf('[poolsize, stride, padding, dataFormat, %s] = %sprepareMaxPool8Args(%s, %s, %s, %s);\n', ...
            NumDimY, getPackageName(nodeTranslation.GraphTranslation), VarPoolsize, VarStride, VarPadding, NumDimX),...
            sprintf('%s = maxpool(%s, poolsize, ''Stride'', stride, ''Padding'', padding, ''DataFormat'', dataFormat);\n', VarY, VarX),...
            ];
    case 2  % New in opset 8
        % Two-output: [Y, Indices] = maxpool(X,POOLSIZE,...)
        Indices  = node.output{2};
        [VarIndices, NumDimIndices] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Indices);
        command = [
            sprintf('%% %s:\n', node.op_type),...
            sprintf('[poolsize, stride, padding, dataFormat, %s, %s] = %sprepareMaxPool8Args(%s, %s, %s, %s);\n',...
            NumDimY, NumDimIndices, getPackageName(nodeTranslation.GraphTranslation), VarPoolsize, VarStride, VarPadding, NumDimX),...
            sprintf('[%s, %s] = maxpool(%s, poolsize, ''Stride'', stride, ''Padding'', padding, ''DataFormat'', dataFormat);\n', VarY, VarIndices, VarX),...
            ];
        nodeTranslation.IntegerOutputTensorNames = string(Indices);
end

nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "prepareMaxPool8Args";
if ismember(X, IntegerTensorNames)
    nodeTranslation.IntegerOutputTensorNames(end+1) = string(Y);
end
end
