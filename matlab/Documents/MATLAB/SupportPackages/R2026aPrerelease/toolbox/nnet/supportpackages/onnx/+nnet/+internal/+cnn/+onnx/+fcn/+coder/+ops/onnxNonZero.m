function [Y, numDimsY] = onnxNonZero(X_, numDimsX_)
%#codegen

% Implements the ONNX NonZero operator

% Copyright 2024 The MathWorks, Inc.
X          = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(X_);
numDimsX   = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);

Coords     = cell(1, max(2,numDimsX));

% Extract the coordinates of the nonzero points 
[Coords{:}] = ind2sub(size(X, 1:max(2,numDimsX)), find(X~=0));

% Concatenate the coordinates into a 2D matrix
Y_ = [Coords{:}];

% Subtract 1 to convert to origin 0
Y_ = Y_ - 1;

if numDimsX < 2
    Y__ = Y_(:,1);
else
    Y__ = Y_;
end

Y = flip(Y__,2);
numDimsY = 2;
end
