function [CeilNode, inits] = createCeilNode(opset, name, input, output)
% A helper function to create a Ceil operator of the specified opset
% version. 

%   Copyright 2024 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
CeilNode = NodeProto;
CeilNode.op_type   = 'Ceil';
CeilNode.name      = name;
CeilNode.input     = input;
CeilNode.output    = output;
inits               = [];
end