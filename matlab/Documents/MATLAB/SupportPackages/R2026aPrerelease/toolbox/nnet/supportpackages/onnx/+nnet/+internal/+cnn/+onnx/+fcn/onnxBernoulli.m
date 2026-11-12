function [y, numDimsY] = onnxBernoulli(X, numDimsX)
% Implements the ONNX Bernoulli operator. 

% Copyright 2024 The MathWorks, Inc.

    numDimsY = numDimsX;
    % rand returns a uniformly distributed random number matrix in (0,1)
    % with size numDimsX. 
    yRand = rand(size(X),"like",X);
    y = double(yRand < X);
end
