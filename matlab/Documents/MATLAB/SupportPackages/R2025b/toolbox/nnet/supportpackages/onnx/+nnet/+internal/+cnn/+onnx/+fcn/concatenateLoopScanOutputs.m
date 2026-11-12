function ConcatenatedScanOutputs = concatenateLoopScanOutputs(ScanOutputs, numDimsScanOutputs)
%   Copyright 2025 The MathWorks, Inc.

% ScanOutputs{j} is a cell array of tensors which should be concatenated to
% produce ConcatenatedScanOutputs{j}. According to the ONNX Doc equivalent
% C-code, "user_defined_vals[j] = user_defined_val", which we interpret to
% mean: "Concatenate ScanOutputs{j} along a new, leading dimension."
%   Because DLT tensors have reversed dimension ordering, this means we
%   should concatenate ScanOutputs{j} along a new, trailing dimension.
ConcatenatedScanOutputs = ScanOutputs;
for j=1:numel(ScanOutputs)
    % Concatenate ScanOutputs{j}
    toConcat = ScanOutputs{j};          % toConcat is a cell array.
    eltNumDims = numDimsScanOutputs{j};
    if eltNumDims > 0
        % Add a trailing dimension to each element to be concatenated
        newEltSize = [size(toConcat{1},1:eltNumDims) 1];
        toConcat = cellfun(@(x)reshape(x, newEltSize), toConcat, 'UniformOutput', false);
    end
    % concatenate along last dimension
    lastDim = eltNumDims+1;
    ConcatenatedScanOutputs{j} = cat(lastDim, toConcat{:});    % dlarray/cat
end
end