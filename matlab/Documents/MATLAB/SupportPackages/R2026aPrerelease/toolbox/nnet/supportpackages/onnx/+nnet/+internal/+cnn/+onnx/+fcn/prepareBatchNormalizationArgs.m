function [offset, scale, datasetMean, datasetVariance, dataFormat, numDimsY, numDimsDatasetMean, numDimsDatasetVariance] = prepareBatchNormalizationArgs(...
    offset, scale, datasetMean, datasetVariance, numDimsX, numDimsDatasetMean, numDimsDatasetVariance)
% Prepares arguments for implementing the ONNX BatchNormalization operator

%   Copyright 2020-2024 The MathWorks, Inc.

offset = dlarray(offset,'C');
scale = dlarray(scale,'C');
if isdlarray(datasetMean)
    datasetMean = extractdata(datasetMean);
end
if isdlarray(datasetVariance)
    datasetVariance = extractdata(datasetVariance);
end
datasetVariance(datasetVariance <= 0) = realmin('single');  % Set nonpositive variance components to a value below eps('single')
dataFormat = [repmat('S', 1, numDimsX-2), 'CB'];
numDimsY = numDimsX;
end
