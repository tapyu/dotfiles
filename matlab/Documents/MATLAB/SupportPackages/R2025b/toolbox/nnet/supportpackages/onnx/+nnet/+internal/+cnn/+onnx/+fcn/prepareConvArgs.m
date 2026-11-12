function [weights, bias, stride, dilationFactor, padding, dataFormat, numDimsY] = prepareConvArgs(...
    weights, bias, stride, dilationFactor, padding, numWtGroups, numDimsX, numDimsW)
% Prepares arguments for implementing the ONNX Conv operator

%   Copyright 2020 The MathWorks, Inc.    

% Weights: The ONNX weight dim is Fcxyz..., where c=C/G, G is numGroups,
% and xyz... are spatial dimensions. DLT "weights" here is the flip of
% that, or ...zyxcF. dlconv requires ...zyxcfG, where f=F/G. So reshape to
% split the last dimension.
sizeW    = size(weights, 1:numDimsW);
F        = sizeW(end);
newWSize = [sizeW(1:numDimsW-1), F/numWtGroups, numWtGroups];
weights  = reshape(weights, newWSize);
% bias
if isempty(bias)
    bias = 0;
end
bias = dlarray(bias(:),'CU');
% Derive missing default attributes from weight tensor
numSpatialDims = numDimsW-2;
if isempty(padding)
    padding = zeros(1, 2*numSpatialDims);
end
if isempty(stride)
    stride = ones(1,numSpatialDims);
end
if isempty(dilationFactor)
    dilationFactor = ones(1,numSpatialDims);
end
% Make the attributes non-dlarrays:
if isa(stride, 'dlarray')
    stride = extractdata(stride);
end
if isa(dilationFactor, 'dlarray')
    dilationFactor = extractdata(dilationFactor);
end
if isa(padding, 'dlarray')
    padding = extractdata(padding);
end
% Make the attributes double row vectors, and flip their dimension ordering
% to reverse-onnx:
stride = fliplr(double(stride(:)'));
dilationFactor = fliplr(double(dilationFactor(:)'));
if isnumeric(padding)       % padding can be "same"
    % ONNX: [x1_begin, ..., xn_begin, x1_end, ...,xn_end]
    % DLT:  [xn_begin, ..., x1_begin;
    %        xn_end, ..., x1_end]       (Note the lrflip and semicolon)
    padding = fliplr(transpose(reshape(padding, [], 2)));
end
% Set dataformat and numdims
dataFormat = [repmat('S', 1, numDimsX-2) 'CB'];
numDimsY = numDimsX;
end
