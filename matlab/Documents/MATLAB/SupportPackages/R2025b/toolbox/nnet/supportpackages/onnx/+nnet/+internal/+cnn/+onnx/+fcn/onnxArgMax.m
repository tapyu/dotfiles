function [Y, numDimsY] = onnxArgMax(X, ONNXAxis, keepdims, numDimsX)
% Implements the ONNX ArgMax operator.

% Copyright 2020-2024 The MathWorks, Inc.
if ONNXAxis < 0
    ONNXAxis = ONNXAxis + numDimsX;
end
DLTAxis = numDimsX - ONNXAxis;
[~, DLTI] = max(X, [], DLTAxis);
Y = DLTI - 1;
if keepdims
    numDimsY = numDimsX;
else
    Y = nnet.internal.cnn.onnx.fcn.onnxSqueeze(Y, ONNXAxis, numDimsX);
    numDimsY = numDimsX - 1; 
end
end
