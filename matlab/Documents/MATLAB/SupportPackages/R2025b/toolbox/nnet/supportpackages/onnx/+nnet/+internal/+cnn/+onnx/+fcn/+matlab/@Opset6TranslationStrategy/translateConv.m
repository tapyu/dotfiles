function nodeTranslation = translateConv(~, nodeTranslation, node, ~)
%   Copyright 2023-2026 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "auto_pad"      "STRING"    true    "NOTSET"
    "dilations"     "INTS"      true    []
    "group"         "INT"       true    1
    "kernel_shape"  "INTS"      true    []
    "pads"          "INTS"      true    []
    "strides"       "INTS"      true    []
    });
% Parse the attributes
[auto_pad, dilations, group, ~, pads, strides] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
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
% 2nd input is W and the 3rd input is B.
if numel(node.input) > 2 && ~isempty(node.input{3})
    BIAS = node.input{3};
    setInitializerType(nodeTranslation.GraphTranslation,'Learnables',BIAS);
    [VarBIAS, ~]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, BIAS);
else
    VarBIAS = "''";
end
% Y = dlconv(X,WEIGHTS,BIAS,...)
Y               = node.output{1};
X               = node.input{1};
setInitializerType(nodeTranslation.GraphTranslation,'Learnables',X);
[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);

WEIGHTS         = node.input{2};
setInitializerType(nodeTranslation.GraphTranslation,'Learnables',WEIGHTS);
[varWeights, numDimsWeights] = getVarAndNumDimName(nodeTranslation.GraphTranslation, WEIGHTS);

StrideName      = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Stride']);
DilationName    = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'DilationFactor']);
PaddingName     = nnet.internal.cnn.onnx.fcn.uniqueName([node.op_type, 'Padding']);
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', StrideName, nnet.internal.cnn.onnx.fcn.RankedArray(strides,1));
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', DilationName, nnet.internal.cnn.onnx.fcn.RankedArray(dilations,1));
setParameters(nodeTranslation.GraphTranslation, 'Nonlearnables', PaddingName, nnet.internal.cnn.onnx.fcn.RankedArray(pads,1));

[VarStrideName, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, StrideName);
[VarDilationName, ~]  = getVarAndNumDimName(nodeTranslation.GraphTranslation, DilationName);
[VarPaddingName, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, PaddingName);

command = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[weights, bias, stride, dilationFactor, padding, dataFormat, %s] = %sprepareConvArgs(%s, %s, %s, %s,%s, %d, %s, %s);\n', ...
    numDimsY, getPackageName(nodeTranslation.GraphTranslation), varWeights, VarBIAS, VarStrideName, VarDilationName, VarPaddingName, ...
    numWtGroups, numDimsX, numDimsWeights),...    
    sprintf('%s = dlconv(%s, weights, bias, ''Stride'', stride, ''DilationFactor'', dilationFactor, ''Padding'', padding, ''DataFormat'', dataFormat);\n', varY, varX),...
    ];

nodeTranslation.MCode                   = string(command);
nodeTranslation.IncludedFunctionNames   = "prepareConvArgs";
end

