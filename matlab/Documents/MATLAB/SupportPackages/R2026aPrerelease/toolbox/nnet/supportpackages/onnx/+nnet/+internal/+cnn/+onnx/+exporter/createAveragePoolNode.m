function [AveragePoolNode, inits] = createAveragePoolNode(~, name, input, output, ...
    poolSize, paddingSize, stride, count_include_pad)
% A helper function to create a AveragePool operator of the specified opset
% version. 

%   Copyright 2024-2025 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
AveragePoolNode = NodeProto;
AveragePoolNode.op_type   = 'AveragePool';
AveragePoolNode.name      = name;
AveragePoolNode.input     = input;
AveragePoolNode.output    = output;
AveragePoolNode.attribute = makeAttributeProto('kernel_shape', 'INTS', poolSize);
if ~isempty(paddingSize)
    AveragePoolNode.attribute = [AveragePoolNode.attribute,...
        makeAttributeProto('pads', 'INTS', paddingSize)];
end
if ~isempty(stride)
    AveragePoolNode.attribute = [AveragePoolNode.attribute,...
        makeAttributeProto('strides', 'INTS', stride)];
end
if ~isempty(count_include_pad)
    AveragePoolNode.attribute = [AveragePoolNode.attribute,...
        makeAttributeProto('count_include_pad', 'INT', count_include_pad)];
end
inits                = [];
end