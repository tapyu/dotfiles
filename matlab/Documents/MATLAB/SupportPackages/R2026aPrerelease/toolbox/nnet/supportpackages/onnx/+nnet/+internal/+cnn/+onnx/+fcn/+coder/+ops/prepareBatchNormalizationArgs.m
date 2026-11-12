function [offset, scale, datasetMean, datasetVariance, dataFormat, numDimsY, numDimsDatasetMean, numDimsDatasetVariance] = prepareBatchNormalizationArgs(...
    offset_, scale_, datasetMean_, datasetVariance_, numDimsX_, numDimsDatasetMean_, numDimsDatasetVariance_)
% Prepares arguments for implementing the ONNX BatchNormalization operator
%#codegen

% Copyright 2024 The MathWorks, Inc.

    offset__                    = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(offset_);
    scale__                     = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(scale_);
    datasetMean                 = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(datasetMean_);
    datasetVariance__           = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(datasetVariance_);
    numDimsX                    = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);
    numDimsDatasetMean          = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsDatasetMean_);
    numDimsDatasetVariance      = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsDatasetVariance_);
    
    offset = dlarray(offset__,'C');
    scale = dlarray(scale__,'C');
    
    datasetVariance = single(datasetVariance__);
    datasetVariance(datasetVariance <= 0) = realmin('single');  % Set nonpositive variance components to a value below eps('single')
    dataFormat = [repmat('S', 1, numDimsX-2), 'CB'];
    numDimsY = numDimsX;
end