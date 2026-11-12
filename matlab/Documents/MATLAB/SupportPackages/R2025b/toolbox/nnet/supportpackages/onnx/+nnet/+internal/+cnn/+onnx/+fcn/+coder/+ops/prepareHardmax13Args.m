function [DLTAxis, numDimsY] = prepareHardmax13Args(ONNXAxis_, numDimsX_)
% Prepares arguments for implementing the ONNX HardMax operator
%#codegen

% Copyright 2024 The MathWorks, Inc.

ONNXAxis   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXAxis_);
numDimsX   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);   

if ONNXAxis<0
    ONNXAxis = ONNXAxis + numDimsX;
end
DLTAxis = numDimsX - ONNXAxis;
numDimsY = numDimsX;
end