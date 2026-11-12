function [Y, numDimsY] = onnxLogSoftmax13(X_, ONNXaxis_, numDimsX_)
% Implements the ONNX LogSoftmax function:
% LogSoftmax(input, axis) = Log(Softmax(input, axis=axis))
% The input is constrained to floating point types.
%#codegen

% Copyright 2026 The MathWorks, Inc.

[Y_, numDimsY] = nnet.internal.cnn.onnx.fcn.coder.ops.onnxSoftmax13(X_, ONNXaxis_, numDimsX_);
Y = log(Y_);
end