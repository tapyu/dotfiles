function [Y, numDimsY] = onnxLogSoftmax13(X, ONNXaxis, numDimsX)
% Implements the ONNX LogSoftmax operator.

% Copyright 2026 The MathWorks, Inc.
[Y, numDimsY] = nnet.internal.cnn.onnx.fcn.onnxSoftmax13(X, ONNXaxis, numDimsX);
Y = log(Y);
end