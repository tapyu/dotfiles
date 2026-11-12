function [poolSize, stride, padding, paddingValue, dataFormat, numDimsY] = prepareAveragePoolArgs(poolSize, stride, padding, count_include_pad, numDimsX)

%   Copyright 2020-2020 The MathWorks, Inc.    

% Prepares arguments for implementing the ONNX AveragePool operator
poolSize    = fliplr(extractdata(poolSize(:)'));
stride      = fliplr(extractdata(stride(:)'));
if isa(padding, 'dlarray')
    padding = extractdata(padding);
end
if isnumeric(padding)
    % ONNX: [x1_begin, ..., xn_begin, x1_end, ...,xn_end]
    % DLT:  [xn_begin, ..., x1_begin;
    %        xn_end, ..., x1_end]       (Note the fliplr and semicolon)
    padding = fliplr(transpose(reshape(padding, [], 2)));
end
if logical(count_include_pad)
    paddingValue = 0;
else
    paddingValue = 'mean';
end
dataFormat  = [repmat('S', 1, numDimsX-2) 'CB'];
numDimsY = numDimsX;
end
