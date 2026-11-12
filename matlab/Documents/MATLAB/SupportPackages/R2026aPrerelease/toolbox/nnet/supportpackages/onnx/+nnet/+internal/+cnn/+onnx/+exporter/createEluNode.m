function [node, inits] = createEluNode(~, name, input, output, Alpha)
% A helper function to create a Elu operator of the specified opset
% version. 

%   Copyright 2025 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
node = NodeProto;
node.op_type   = 'Elu';
node.name      = name;
node.input     = input;
node.output    = output;
node.attribute = makeAttributeProto('alpha', 'FLOAT', Alpha);
inits          = [];
end