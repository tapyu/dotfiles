function [Y, numDimsY] = onnxHardSigmoid(X, alpha,beta, numDimsX)
% Implements the ONNX HardSigmoid operator

% Copyright 2024 The MathWorks, Inc.

numDimsY = numDimsX;
Y = max(0,min(1,alpha*X+beta));
end
