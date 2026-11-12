function [ConcatNode, inits] = createConcatNode(opset, name, input, output, axis)
% A helper function to create a Concat operator of the specified opset
% version. 

%   Copyright 2022-2025 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
ConcatNode = NodeProto;
ConcatNode.op_type   = 'Concat';
ConcatNode.name      = name;
ConcatNode.input     = input;
ConcatNode.output    = output;
ConcatNode.attribute =  makeAttributeProto('axis', 'INT', axis);
inits                = [];
end