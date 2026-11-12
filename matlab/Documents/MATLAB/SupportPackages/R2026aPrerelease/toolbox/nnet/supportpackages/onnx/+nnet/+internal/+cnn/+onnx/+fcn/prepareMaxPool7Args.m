function [poolsize, stride, padding, dataFormat, numDimsY] = prepareMaxPool7Args(poolsize, stride, padding, numDimsX)
% Prepares arguments for implementing the ONNX MaxPool-7 operator

%   Copyright 2020 The MathWorks, Inc.    

poolsize    = fliplr(extractdata(poolsize(:)'));
stride      = fliplr(extractdata(stride(:)'));
% padding
if isa(padding, 'dlarray')
    padding = extractdata(padding);
end
if isnumeric(padding)
    % ONNX: [x1_begin, ..., xn_begin, x1_end, ...,xn_end]
    % DLT:  [xn_begin, ..., x1_begin;
    %        xn_end, ..., x1_end]       (Note the lrflip and semicolon)
    padding = fliplr(transpose(reshape(padding, [], 2)));
end
dataFormat  = [repmat('S', 1, numDimsX-2) 'CB'];
numDimsY    = numDimsX;
end
