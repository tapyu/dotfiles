function [MaxPoolNode, inits] = createMaxPoolNode(~, name, input, output,...
    kernel_shape, pads, strides)
% varargin in
% A helper function to create a MaxPoolNode operator of the specified opset
% version. 

%   Copyright 2024-2025 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
MaxPoolNode = NodeProto;
MaxPoolNode.op_type   = 'MaxPool';
MaxPoolNode.name      = name;
MaxPoolNode.input     = input;
MaxPoolNode.output    = output;
MaxPoolNode.attribute = makeAttributeProto('kernel_shape', 'INTS', kernel_shape);
if ~isempty(pads)
    MaxPoolNode.attribute = [MaxPoolNode.attribute, makeAttributeProto('pads', 'INTS', pads)];
end
if ~isempty(strides)
    MaxPoolNode.attribute = [MaxPoolNode.attribute,makeAttributeProto('strides', 'INTS', strides)];
end
inits                = [];
end