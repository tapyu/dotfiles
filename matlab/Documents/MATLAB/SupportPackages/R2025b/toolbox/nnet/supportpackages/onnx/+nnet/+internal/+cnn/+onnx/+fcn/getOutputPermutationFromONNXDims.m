function perm = getOutputPermutationFromONNXDims(onnxNDims)

%   Copyright 2020 The MathWorks, Inc.    

if onnxNDims == 2
    % Do nothing
    perm = [1 2];
elseif onnxNDims == 3
    % Convert ONNX SNC -> MATLAB CNS
    perm = [3 2 1];
elseif onnxNDims == 4
    % Convert ONNX WHCN -> MATLAB HWCN
    perm = [2 1 3 4];
elseif onnxNDims == 5
    % Convert ONNX DWHCN -> MATLAB HWDCN
    perm = [3 2 1 4 5];
elseif onnxNDims == 6
    % Convert ONNX DWHCNS -> MATLAB HWDCNS
    perm = [3 2 1 4 5 6];
end

end