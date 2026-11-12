function [nodeProto,parameterInitializers] = createZScoreNodes(opsetVersion, inputImageTensorName, permIdx, layer)
    % This utility creates a series of nodes and initializers that will
    % perform z-score normalization to the input: 
    % Y = (X - Mean) ./ StandardDeviation

    %   Copyright 2019-2025 The MathWorks, Inc.

    import nnet.internal.cnn.onnx.*
    Mean = layer.Mean; 
    StandardDeviation = layer.StandardDeviation; 

    % Sub node to subtract the mean
    subNodeName             = [inputImageTensorName '_Sub'];
    meanName                = [inputImageTensorName '_Mean'];
    subNodeProto            = NodeProto;
    subNodeProto.op_type	= 'Sub';
    subNodeProto.name       = subNodeName;
    subNodeProto.input      = {inputImageTensorName, meanName};
    subNodeProto.output     = {subNodeName};
    if opsetVersion < 7
        subNodeProto.attribute = [...
            makeAttributeProto('broadcast', 'INT', 1),...
            makeAttributeProto('axis', 'INT', 1)];
    end

    % Sub node parameter initializer
    meanTensor              = TensorProto;
    meanTensor.name         = meanName;
    meanTensor.data_type    = TensorProto_DataType.FLOAT;
    permutedMean            = permute(Mean, permIdx); % NNT is hwc/hwdc, onnx is chw/chwd (with n=1 and broadcast)
    meanTensor.raw_data     = rawData(single(permutedMean));
    meanTensor.dims         = dimVector(size(permutedMean), numel(permIdx));% dims = NCHW or NCHWD 
    if opsetVersion < 7
        meanTensor.dims     = meanTensor.dims(2:end);% dims = CHW or CHWD
    end

    % Divide by standard deviation to get Z-Score
    divNodeName             = [inputImageTensorName '_Div']; 
    standardDeviationName   = [inputImageTensorName '_StandardDeviation']; 
    divNodeProto            = NodeProto;
    divNodeProto.op_type    = 'Div';
    divNodeProto.name       = divNodeName;
    divNodeProto.input      = {subNodeName, standardDeviationName};
    divNodeProto.output     = {divNodeName};
    if opsetVersion < 7
        divNodeProto.attribute = [...
            makeAttributeProto('broadcast', 'INT', 1),...
            makeAttributeProto('axis', 'INT', 1)];
    end
    
    % Div node parameter initializer 
    standardDeviationTensor            = TensorProto;
    standardDeviationTensor.name       = standardDeviationName;
    standardDeviationTensor.data_type  = TensorProto_DataType.FLOAT;
    permutedStandardDev                = permute(StandardDeviation, permIdx);% NNT is hwc/hwdc, onnx is chw/chwd (with n=1 and broadcast)
    standardDeviationTensor.raw_data   = rawData(single(permutedStandardDev));
    standardDeviationTensor.dims       = dimVector(size(permutedStandardDev), numel(permIdx));% dims = NCHW or NCHWD 
    if opsetVersion < 7
        standardDeviationTensor.dims   = standardDeviationTensor.dims(2:end);% dims = CHW or CHWD
    end

    % Finish up 
    nodeProto                          = [subNodeProto divNodeProto];
    parameterInitializers              = [meanTensor standardDeviationTensor];

end
