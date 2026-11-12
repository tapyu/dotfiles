function [splitNode, inits] = createSpaceToDepthNode(~, name, input, output, blocksize)
% A helper function to create a SpaceToDepth operator of the specified opset
% version. 

%   Copyright 2024-2025 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
splitNode            = NodeProto;
splitNode.op_type    = 'SpaceToDepth';
splitNode.name       = name;
splitNode.input      = input;
splitNode.output     = output;
splitNode.attribute = makeAttributeProto('blocksize', 'INT', blocksize);
inits                  = [];
end
