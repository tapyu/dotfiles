function [dim1, dim2, origSize, numDimsX] = prepareHardmaxArgs(X_, ONNXAxis_, numDimsX_)
% Prepares arguments for implementing the ONNX HardMax operator
%#codegen

% Copyright 2024 The MathWorks, Inc.

X          = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(X_);
ONNXAxis   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXAxis_);
numDimsX   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);

if ONNXAxis<0
    ONNXAxis = ONNXAxis + numDimsX;
end
dim2     = prod(size(X, numDimsX+1-ONNXAxis:numDimsX));     % numel on the right
dim1     = numel(X)/dim2;                                   % numel on the left
origSize = size(X);
end
