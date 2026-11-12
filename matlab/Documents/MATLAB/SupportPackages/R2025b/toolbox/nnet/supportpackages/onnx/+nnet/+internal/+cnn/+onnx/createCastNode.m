function [CastNode, inits] = createCastNode(opset, name, input, output, to)
% A helper function to create a Cast operator of the specified opset
% version. 

%   Copyright 2024 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
CastNode = NodeProto;
CastNode.op_type   = 'Cast';
CastNode.name      = name;
CastNode.input     = input;
CastNode.output    = output;
CastNode.attribute = [...
        makeAttributeProto('to', 'INT', to)];    
inits               = [];
end