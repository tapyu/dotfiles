function [Y, numDimsY] = onnxSoftmax13(X, ONNXaxis, numDimsX)
% Implements the ONNX Softmax function:
% Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
% The input is constrained to floating point types.

% Copyright 2021-2024 The MathWorks, Inc.

if ONNXaxis < 0
    ONNXaxis = ONNXaxis + numDimsX;
end
DLTaxis = numDimsX - ONNXaxis;

X = X - max(X, [], DLTaxis); % Subtract max(X) for numerical stability
expX = exp(X);
dims = nnet.internal.cnn.onnx.fcn.prepareReduceArgs(ONNXaxis, numDimsX);
Y = expX ./ sum(expX, dims);
numDimsY = numDimsX;

end