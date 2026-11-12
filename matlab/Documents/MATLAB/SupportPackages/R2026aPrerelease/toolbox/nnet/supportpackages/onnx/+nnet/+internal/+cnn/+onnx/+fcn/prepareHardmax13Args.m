function [DLTAxis, numDimsY] = prepareHardmax13Args(ONNXAxis, numDimsX)
% Prepares arguments for implementing the ONNX HardMax operator

%   Copyright 2021 The MathWorks, Inc.    

if ONNXAxis<0
    ONNXAxis = ONNXAxis + numDimsX;
end
DLTAxis = numDimsX - ONNXAxis;
numDimsY = numDimsX;
end