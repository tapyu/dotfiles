function perm = getInputPermutationFromONNXDims(onnxNDims)

%   Copyright 2020 The MathWorks, Inc.    

if onnxNDims == 4
    % Convert MATLAB HWCN -> NCHW
    perm = [4 3 1 2];
end
end
