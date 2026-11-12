function [Y, numDimsY] = onnxArgMax(X_, ONNXAxis_, keepdims_, numDimsX_)
% Implements the ONNX ArgMax operator.
%#codegen

% Copyright 2024 The MathWorks, Inc.

X          = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(X_);
ONNXAxis   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXAxis_);
keepdims   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(keepdims_);
numDimsX   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);


if ONNXAxis < 0
    ONNXAxis = ONNXAxis + numDimsX;
end

DLTAxis = numDimsX - ONNXAxis;
[~, DLTI] = max(X, [], DLTAxis);
Y_ = DLTI - 1;

if keepdims
    Y = Y_;
    numDimsY = numDimsX;
else
    Y = nnet.internal.cnn.onnx.fcn.coder.ops.onnxSqueeze(Y_, ONNXAxis, numDimsX);
    numDimsY = numDimsX - 1; 
end
end