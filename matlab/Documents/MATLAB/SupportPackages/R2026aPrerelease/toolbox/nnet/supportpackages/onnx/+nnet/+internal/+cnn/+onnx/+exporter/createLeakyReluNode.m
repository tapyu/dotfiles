function [LeakyReluNode, inits] = createLeakyReluNode(~, name, input, output, alpha)
% A helper function to create a LeakyRelu operator of the specified opset
% version. 

%   Copyright 2025 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
LeakyReluNode = NodeProto;
LeakyReluNode.op_type   = 'LeakyRelu';
LeakyReluNode.name      = name;
LeakyReluNode.input     = input;
LeakyReluNode.output    = output;
LeakyReluNode.attribute  = makeAttributeProto('alpha', 'FLOAT', alpha);
inits                = [];
end