function [dim1, dim2, origSize, numDimsX] = prepareSoftmaxArgs(X, ONNXAxis, numDimsX)
% Prepares arguments for implementing the ONNX Softmax operator

%   Copyright 2020 The MathWorks, Inc.    

if ONNXAxis<0
    ONNXAxis = ONNXAxis + numDimsX;
end
dim2     = prod(size(X, numDimsX+1-ONNXAxis:numDimsX));   % numel on the right
dim1     = numel(X)/dim2;                                 % numel on the left
origSize = size(X);
end
