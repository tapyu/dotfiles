function nodeTranslation = translateBatchNormalization(~, nodeTranslation, node, ~)
%   Copyright 2024-2026 The MathWorks, Inc.

% Define the legal attributes. Table columns are: onnxName, type, isOptional, default.
% To see legal type strings: string(enumeration('nnet.internal.cnn.onnx.AttributeProto_AttributeType'))
AttributeTable = cell2table({
    "epsilon"      "FLOAT"      true    1e-5
    "momentum"     "FLOAT"  	true    0.9
    "spatial"      "INT"        true    1
    "is_test"      "INT"        true    0
    });
% Parse the attributes
[Epsilon, ~, spatial, ~] = nnet.internal.cnn.onnx.parseNodeAttributes(node, AttributeTable);
if Epsilon <= 0
    Epsilon = 1e-5;
    nodeTranslation.Issues = [nodeTranslation.Issues nnet.internal.cnn.onnx.NodeTranslationWarning(node,...
        message('nnet_cnn_onnx:onnx:BadEpsilon'))];
end
Epsilon = double(Epsilon);
if spatial == 0
    nodeTranslation.Issues = [nodeTranslation.Issues nnet.internal.cnn.onnx.NodeTranslationWarning(node,...
        message('nnet_cnn_onnx:onnx:PerChannelMean'))];
end

ScaleName       = node.input{2};
OffsetName      = node.input{3};
meanName        = node.input{4};
varName         = node.input{5};
% For coder, mean and variance are inference-only constants (not State
% properties, which block codegen). Let getVarAndNumDimName resolve them
% as Nonlearnables via the isInitializer fallback path.
setInitializerType(nodeTranslation.GraphTranslation,'Learnables',ScaleName);
setInitializerType(nodeTranslation.GraphTranslation,'Learnables',OffsetName);
if isInitializer(nodeTranslation.GraphTranslation, meanName)
    [~,data] = getInitializerValue(nodeTranslation.GraphTranslation, meanName);
    nodeTranslation.GraphTranslation.InitialState.(meanName) = data;
end
if isInitializer(nodeTranslation.GraphTranslation, varName)
    [~,data] = getInitializerValue(nodeTranslation.GraphTranslation, varName);
    nodeTranslation.GraphTranslation.InitialState.(varName) = data;
end
[varScale, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, ScaleName);
[varOffset, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, OffsetName);
[varMean, numDimsMean] = getVarAndNumDimName(nodeTranslation.GraphTranslation, meanName);
[varVar, numDimsVar] = getVarAndNumDimName(nodeTranslation.GraphTranslation, varName);

% Coder only supports inference (Training=false), so emit only the
% inference branch of batchnorm (no state update).

Y               = node.output{1};
X               = node.input{1};
[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
DATASETMEAN     = varMean;
DATASETVARIANCE = varVar;
OFFSET          = varOffset;
SCALE           = varScale;

offsetTempName          = nnet.internal.cnn.onnx.fcn.uniqueCoderName('offset');
scaleTempName           = nnet.internal.cnn.onnx.fcn.uniqueCoderName('scale');
datasetMeanTempName     = nnet.internal.cnn.onnx.fcn.uniqueCoderName('datasetMean');
datasetVarianceTempName = nnet.internal.cnn.onnx.fcn.uniqueCoderName('datasetVariance');
dataFormatTempName      = nnet.internal.cnn.onnx.fcn.uniqueCoderName('dataFormat');

xTempName      = nnet.internal.cnn.onnx.fcn.uniqueCoderName('X');
yTempName      = nnet.internal.cnn.onnx.fcn.uniqueCoderName('Y');

code = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[%s, %s, %s, %s, %s, %s, %s, %s] = %sprepareBatchNormalizationArgs(%s, %s, %s, %s, %s, %s, %s);\n',...
    offsetTempName, scaleTempName, datasetMeanTempName, datasetVarianceTempName, dataFormatTempName,...
    numDimsY, '~', '~', getPackageName(nodeTranslation.GraphTranslation), ...
    OFFSET, SCALE, DATASETMEAN, DATASETVARIANCE, numDimsX, numDimsMean, numDimsVar),...
    sprintf('%s = dlarray(%sextractIfDlarray(%s));\n', xTempName, getPackageName(nodeTranslation.GraphTranslation), varX),...
    sprintf('%s = batchnorm(%s, %s, %s, %s, %s, ''Epsilon'', %f, ''DataFormat'', %s);\n', yTempName, xTempName,...
    offsetTempName, scaleTempName, datasetMeanTempName, datasetVarianceTempName, Epsilon, dataFormatTempName),...
    sprintf('%s = %sextractIfDlarray(%s);\n', varY, getPackageName(nodeTranslation.GraphTranslation), yTempName),...
    ];
nodeTranslation.MCode                   = string(code);
nodeTranslation.IncludedFunctionNames   = "prepareBatchNormalizationArgs";
end

