function [Y, numDimsY, Mask, numDimsMask] = onnxDropout12(X, seed, ratio, Training, numDimsX)
% Implements the ONNX Dropout-12 operator. 

% Copyright 2021 The MathWorks, Inc.

originalSeed = [];
if ~isempty(seed)
    originalSeed = setSeed(seed, X);
end
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
% Reset the seed
if ~isempty(originalSeed)
    setSeed(originalSeed,X);
end

    function oldSeed = setSeed(seedVal, X)
        % Set the seed for the random number generator
        if isgpuarray(X)
            oldSeed = gpurng;
            gpurng(seedVal);
        else
            oldSeed = rng;
            rng(seedVal);
        end
    end

end
