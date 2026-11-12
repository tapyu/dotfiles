function [FlattenNode, inits] = createFlattenNode(~, name, input, output, axis)
% A helper function to create a Flatten operator of the specified opset
% version. 

%   Copyright 2025 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
FlattenNode = NodeProto;
FlattenNode.op_type   = 'Flatten';
FlattenNode.name      = name;
FlattenNode.input     = input;
FlattenNode.output    = output;
FlattenNode.attribute  = makeAttributeProto('axis', 'INT', axis);
inits                = [];
end