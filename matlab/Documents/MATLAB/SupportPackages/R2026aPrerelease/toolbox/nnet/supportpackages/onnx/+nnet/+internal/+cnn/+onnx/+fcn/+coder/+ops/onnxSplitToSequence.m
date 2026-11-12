function [output_sequence, output_sequenceNumDims] = onnxSplitToSequence(X_, ONNXAxis_, split_, keepdims_, numDimsX_, numDimsSplit_)
% Implements the ONNX SplitToSequence operator
%#codegen
% Copyright 2024-2025 The MathWorks, Inc.
    X               = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(X_);
    ONNXAxis        = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(ONNXAxis_);
    split           = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(split_);
    keepdims        = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(keepdims_);
    numDimsX        = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);
    numDimsSplit    = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsSplit_);
    
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
        % 'split' is the size of each chunk. 
        if isdlarray(split)
            split__ = coder.const(extractdata(split));
        else
            split__ = coder.const(split);
        end
        numberOfchunks        = floor(axisSize/split__);        
        if rem(axisSize, split__) ~= 0
            splits      = repmat(split__, 1, numberOfchunks+1);    % Fill all but last
            splits(end)	= axisSize - split__ * numberOfchunks; % Fill last if there's remainder
        else
            splits      = repmat(split__, 1, numberOfchunks);    % Fill all but last
        end
        keepdims    = 1;
    else
        % 'split' is 1D and gives the sizes
        splits      = split;
        keepdims    = 1;
    end
    
    % Do the splitting
    S = cell(1, numDimsX); % Important to use numDimsX. ndims(X) may be too small.
    coder.unroll();
    for j = 1:numDimsX
        S{j} = 1 : size(X,j);
    end        
    
    splitIndices = [0 cumsum(splits(:)')];
    numY = numel(splitIndices)-1;
    
    if keepdims
        numDimsY = numDimsX;
    else
        numDimsY = numDimsX-1;
    end
    
    output_sequence = cell(1, 2*numY);
    coder.unroll();
    for i = 1:numY
        from                = splitIndices(i) + 1;
        to                  = splitIndices(i+1);
        S{DLTAxis}          = from:to;
        % The first numY outputs are the Y's. The second numY outputs are their
        % numDims.
        output_sequence_i	= X(S{:});
        if ~keepdims
            output_sequence{i} = nnet.internal.cnn.onnx.fcn.coder.ops.onnxSqueeze(output_sequence_i, ONNXAxis, numDimsX);
        else
            output_sequence{i} = output_sequence_i;
        end
        output_sequence{i + numY} = numDimsY;
    end
    output_sequenceNumDims = 1;     % NumDims of the 1D sequence itself.
end