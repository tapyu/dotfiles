function [softmaxNode, inits] = createSoftmaxNode(~, name, input, output, axis)
% A helper function to create a Softmax operator of the specified opset
% version.

%   Copyright 2025 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
softmaxNode                 = NodeProto;
softmaxNode.op_type         = 'Softmax';
softmaxNode.name            = name;
softmaxNode.input           = input;
softmaxNode.output          = output;
softmaxNode.attribute       = makeAttributeProto('axis', 'INT', axis);
inits                       = [];
end