function [y, numDimsY] = onnxBernoulli(X, numDimsX_)
% Implements the ONNX Bernoulli operator. 
%#codegen

% Copyright 2024 The MathWorks, Inc.

    numDimsX   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);
    numDimsY = numDimsX;
    % rand returns a uniformly distributed random number matrix in (0,1)
    % with size numDimsX. 
    yRand = rand(size(X));
    y = yRand < X;
end
