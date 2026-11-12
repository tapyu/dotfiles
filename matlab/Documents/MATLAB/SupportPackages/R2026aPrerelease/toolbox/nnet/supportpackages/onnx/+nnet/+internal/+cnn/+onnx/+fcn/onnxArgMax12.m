function [Y, numDimsY] = onnxArgMax12(X, ONNXAxis, keepdims, selectLastIndex, numDimsX)
% Implements the ONNX ArgMax operator.

% Copyright 2021-2024 The MathWorks, Inc.

if ONNXAxis < 0
    ONNXAxis = ONNXAxis + numDimsX;
end
DLTAxis = numDimsX - ONNXAxis;
if selectLastIndex % New in Opset 12
    X2 = flip(X, DLTAxis);
    [~, DLTI] = max(X2, [], DLTAxis);
    Y = size(X, DLTAxis) - DLTI;
else
    [~, DLTI] = max(X, [], DLTAxis);
    Y = DLTI - 1;
end
if keepdims
    numDimsY = numDimsX;
else
    Y = nnet.internal.cnn.onnx.fcn.onnxSqueeze(Y, ONNXAxis, numDimsX);
    numDimsY = numDimsX - 1; 
end
end
