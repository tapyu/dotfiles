function [Y, numDimsY, Mask, numDimsMask] = onnxDropout7(X, ratio, Training, numDimsX)
% Implements the ONNX Dropout-7 operator. 

% Copyright 2020 The MathWorks, Inc.

if Training
    Scale = 1/(1 - ratio);
    Mask = rand(size(X),'like',X) >= ratio;
    Y = Scale .* X .* Mask;
else
    Y = X;
    Mask = ones(size(X),'like',X);
end
numDimsY = numDimsX;
numDimsMask = numDimsX;
end
