function [X, numDimsX] = onnxElu(X, alpha, numDimsX)
% Implements the ONNX Elu operator

% Copyright 2021 The MathWorks, Inc.

X(X<=0) = alpha*(exp(X(X<=0))-1);
end
