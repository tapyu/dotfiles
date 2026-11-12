function [node, inits] = createMaxRoiPoolNode(~, name, input, output, OutputSize, ScaleFactor)
% A helper function to create a MaxRoiPool operator of the specified opset
% version. 

%   Copyright 2025 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
node = NodeProto;
node.op_type   = 'MaxRoiPool';
node.name      = name;
node.input     = input;
node.output    = output;
node.attribute = [...
                makeAttributeProto('pooled_shape',  'INTS', OutputSize),...
                makeAttributeProto('spatial_scale', 'FLOAT', ScaleFactor),...
                ];
inits          = [];
end