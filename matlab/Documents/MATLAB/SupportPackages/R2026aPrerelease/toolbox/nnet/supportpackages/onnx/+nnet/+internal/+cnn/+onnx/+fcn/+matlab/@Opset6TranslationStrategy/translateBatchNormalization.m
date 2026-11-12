function nodeTranslation = translateBatchNormalization(~, nodeTranslation, node, ~)
%   Copyright 2023-2026 The MathWorks, Inc.

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
if isInitializer(nodeTranslation.GraphTranslation, meanName)
    [~,data] = getInitializerValue(nodeTranslation.GraphTranslation, meanName);
    nodeTranslation.GraphTranslation.InitialState.(meanName) = data;
end
if isInitializer(nodeTranslation.GraphTranslation, varName)
    [~,data] = getInitializerValue(nodeTranslation.GraphTranslation, varName);
    nodeTranslation.GraphTranslation.InitialState.(varName) = data;
end
setInitializerType(nodeTranslation.GraphTranslation,'Learnables',ScaleName);
setInitializerType(nodeTranslation.GraphTranslation,'Learnables',OffsetName);
[varScale, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, ScaleName);
[varOffset, ~] = getVarAndNumDimName(nodeTranslation.GraphTranslation, OffsetName);
[varMean, numDimsMean] = getVarAndNumDimName(nodeTranslation.GraphTranslation, meanName);
[varVar, numDimsVar] = getVarAndNumDimName(nodeTranslation.GraphTranslation, varName);
% The call to Batchnorm is followed by code that updates state variables:
% [Y,DATASETMEAN,DATASETVARIANCE] = batchnorm(X,OFFSET,SCALE,DATASETMEAN,DATASETVARIANCE,'DataFormat',FMT)

Y               = node.output{1};
X               = node.input{1};
[varX, numDimsX] = getVarAndNumDimName(nodeTranslation.GraphTranslation, X);
[varY, numDimsY] = getVarAndNumDimName(nodeTranslation.GraphTranslation, Y);
DATASETMEAN     = varMean;
DATASETVARIANCE = varVar;
OFFSET          = varOffset;
SCALE           = varScale;
code = [
    sprintf('%% %s:\n', node.op_type),...
    sprintf('[offset, scale, datasetMean, datasetVariance, dataFormat, %s, %s, %s] = %sprepareBatchNormalizationArgs(%s, %s, %s, %s, %s, %s, %s);\n',...
    numDimsY, numDimsMean, numDimsVar, getPackageName(nodeTranslation.GraphTranslation), ...
    OFFSET, SCALE, DATASETMEAN, DATASETVARIANCE, numDimsX, numDimsMean, numDimsVar),...
    sprintf('if Training\n'),...
    sprintf('[%s, %s, %s] = batchnorm(%s, offset, scale, datasetMean, datasetVariance, ''Epsilon'', %f, ''DataFormat'', dataFormat);\n', ...
    varY, DATASETMEAN, DATASETVARIANCE, varX, Epsilon),...
    sprintf('else\n'),...
    sprintf('%s = batchnorm(%s, offset, scale, datasetMean, datasetVariance, ''Epsilon'', %f, ''DataFormat'', dataFormat);\n', varY, varX, Epsilon),...
    sprintf('end\n')];
nodeTranslation.MCode                   = string(code);
nodeTranslation.IncludedFunctionNames   = "prepareBatchNormalizationArgs";
end

