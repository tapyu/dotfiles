function [Y, numDimsY] = onnxArgMax12(X, ONNXAxis_, keepdims_, selectLastIndex_, numDimsX_)
% Implements the ONNX ArgMax operator.
%#codegen

% Copyright 2024 The MathWorks, Inc.

ONNXAxis   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXAxis_);
keepdims   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(keepdims_);
selectLastIndex   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(selectLastIndex_);
numDimsX   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);

if ONNXAxis < 0
    ONNXAxis = ONNXAxis + numDimsX;
end

DLTAxis = numDimsX - ONNXAxis;

if selectLastIndex % New in Opset 12
    X2 = flip(X, DLTAxis);
    [~, DLTI] = max(X2, [], DLTAxis);
    Y_ = size(X, DLTAxis) - DLTI;
else
    [~, DLTI] = max(X, [], DLTAxis);
    Y_ = DLTI - 1;
end
if keepdims
    Y = Y_;
    numDimsY = numDimsX;
else
    Y = nnet.internal.cnn.onnx.fcn.coder.ops.onnxSqueeze(Y_, ONNXAxis, numDimsX);
    numDimsY = numDimsX - 1; 
end
end