function [GatherNode, inits] = createGatherNode(opset, name, input, output, axis)
% A helper function to create a Gather operator of the specified opset
% version. 

%   Copyright 2024 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
GatherNode = NodeProto;
GatherNode.op_type   = 'Gather';
GatherNode.name      = name;
GatherNode.input     = input;
GatherNode.output    = output;
GatherNode.attribute = makeAttributeProto('axis', 'INT', axis);
inits                = [];
end