function [weights, bias, stride, dilation, cropping, dataFormat, numDimsY] = prepareConvTransposeArgs(ONNXWeights, bias, stride, dilation, pads, auto_pad, output_shape, numWtGroups, sizeX, numDimsX, numDimsW)
% Prepares arguments for implementing the ONNX ConvTranspose operator

%   Copyright 2020-2021 The MathWorks, Inc.    

% The ONNX weight dim is Cfhwd..., where f=F/G, G is numGroups and hwd...
% are the spatial dimensions of the kernel. The input argumnt "ONNXWeights"
% here is the flip of that, or ...dwhfC. We want ...dwhfcG, where c=C/G. So
% we just need to perform a reshape to split the last dimension from C to
% cG.
sizeW       = size(ONNXWeights, 1:numDimsW);
C           = sizeW(end);
newWSize    = [sizeW(1:numDimsW-1), C/numWtGroups, numWtGroups];
weights     = reshape(ONNXWeights, newWSize);
if isempty(bias)
    bias = 0;
end
bias = dlarray(bias(:),'CU');
% Derive missing default attributes from weight tensor
numSpatialDims = numDimsW-2;
if isempty(pads)
    pads = zeros(1, 2*numSpatialDims);
end
if isempty(stride)
    stride = ones(1,numSpatialDims);
end
if isempty(dilation)
    dilation = ones(1,numSpatialDims);
end
% Make the attributes non-dlarrays:
if isa(stride, 'dlarray')
    stride = extractdata(stride);
end
if isa(dilation, 'dlarray')
    dilation = extractdata(dilation);
end
if isa(pads, 'dlarray')
    pads = extractdata(pads);
end
if isa(output_shape, 'dlarray')
    output_shape = extractdata(output_shape);
end
if ~isempty(output_shape)
    % Process output_shape. The ONNX Doc says:
    % "If output_shape is specified pads values are ignored."
    % And then 'pads' is recalculated as follows:
    % total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
    % If (auto_pads == SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
    % Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).

    % sizeX is reverse-onnx with trailing singletons dropped. ...zyxCN.
    % Extract the ONNX spatial shape as 'input_size', and put it in forward
    % ONNX ordering.
    if numDimsX > numel(sizeX)
        sizeX = [sizeX, ones(1,numDimsX-numel(sizeX))];     % sizeX is now ...zyxCN
    end
    input_size = fliplr(sizeX(1:numSpatialDims));     % xyz...
    % Recover the kernel_shape from the weights in forward ONNX ordering
    kernel_shape = fliplr(sizeW(1:numSpatialDims));
    % Calculate the new pads value in forward ONNX ordering
    output_padding = 0;             % output_padding is not supported (yet).
    D = numel(input_size);
    for i = 1:D
        total_padding(i) = stride(i) * (input_size(i) - 1) + output_padding + ((kernel_shape(i) - 1) * dilation(i) + 1) - output_shape(i);
        start_i = i;
        end_i = D+i;
        if auto_pad=="SAME_UPPER"
            pads(start_i) = floor(total_padding(i)/2);
            pads(end_i) = total_padding(i) - floor(total_padding(i)/2);
        else
            pads(start_i) = total_padding(i) - floor(total_padding(i)/2);
            pads(end_i) = floor(total_padding(i)/2);
        end
    end
end
% Make the attributes double row vectors, and flip their dimension ordering
% to reverse-onnx:
stride = fliplr(double(stride(:)'));
dilation = fliplr(double(dilation(:)'));
if isnumeric(pads)       % padding can be "same"
    % ONNX: [x1_begin, ..., xn_begin, x1_end, ...,xn_end]
    % DLT:  [xn_begin, ..., x1_begin;
    %        xn_end, ..., x1_end]       (Note the left-right flip and the semicolon)
    pads = fliplr(transpose(reshape(pads, [], 2)));
end
cropping = pads;
dataFormat  = [repmat('S', 1, numDimsX-2) 'CB'];
numDimsY = numDimsX;
end
