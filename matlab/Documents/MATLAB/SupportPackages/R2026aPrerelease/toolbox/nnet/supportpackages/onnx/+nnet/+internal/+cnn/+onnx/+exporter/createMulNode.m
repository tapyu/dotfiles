function [MulNode, inits] = createMulNode(opset, name, input, output)
% A helper function to create a Mul operator of the specified opset
% version. 

%   Copyright 2025 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
MulNode            = NodeProto;
MulNode.op_type    = 'Mul';
MulNode.name       = name;
MulNode.input      = input;
MulNode.output     = output;
inits = [];
if opset < 7
    MulNode.attribute = makeAttributeProto('broadcast', 'INT', 1);
end
end
