function [output_sequence, output_sequenceNumDims] = onnxSplitToSequence(X, ONNXAxis, split, keepdims, numDimsX, numDimsSplit)
% Implements the ONNX SplitToSequence operator

% Copyright 2020-2025 The MathWorks, Inc.

% ONNXaxis is origin 0.
if ONNXAxis<0
    ONNXAxis = ONNXAxis + numDimsX;
end
DLTAxis = numDimsX - ONNXAxis;
% Set 'splits' to be the split sizes
axisSize = size(X, DLTAxis);
if isempty(split)
    % Split them all
    splits      = ones(1, axisSize);
elseif numDimsSplit==0
    % 'split' is the number of splits
    if isdlarray(split)
        split = extractdata(split);
    end
    numberOfchunks        = floor(axisSize/split);        
    if rem(axisSize, split) ~= 0
        splits      = repmat(split, 1, numberOfchunks+1);    % Fill all but last
        splits(end)	= axisSize - split * numberOfchunks; % Fill last if there's remainder
    else
        splits      = repmat(split, 1, numberOfchunks);    % Fill all but last
    end
    keepdims    = 1;
else
    % 'split' is 1D and gives the sizes
    if isdlarray(split)
        split = extractdata(split);
    end
    splits      = split;
    keepdims    = 1;
end
% Do the splitting
S = repmat({':'}, 1, numDimsX);        % Important to use numDimsX. ndims(X) may be too small.
splitIndices = [0 cumsum(splits(:)')];
numY = numel(splitIndices)-1;
if keepdims
    numDimsY = numDimsX;
else
    numDimsY = numDimsX-1;
end
for i = 1:numY
    from                = splitIndices(i) + 1;
    to                  = splitIndices(i+1);
    S{DLTAxis}     = from:to;
    % The first numY outputs are the Y's. The second numY outputs are their
    % numDims.
    output_sequence{i}	= X(S{:});
    if ~keepdims
        output_sequence{i} = nnet.internal.cnn.onnx.fcn.onnxSqueeze(output_sequence{i}, ONNXAxis, numDimsX);
    end
    output_sequence{i + numY} = numDimsY;
end
output_sequenceNumDims = 1;     % NumDims of the 1D sequence itself.
end
