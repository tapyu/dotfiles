function [node, inits] = createLRNNode(~, name, input, output, Alpha, Beta, K, WindowChannelSize)
% A helper function to create a LRN operator of the specified opset
% version. 

%   Copyright 2025 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
node = NodeProto;
node.op_type   = 'LRN';
node.name      = name;
node.input     = input;
node.output    = output;
node.attribute = [...
                makeAttributeProto('alpha',	'FLOAT', Alpha),...
                makeAttributeProto('beta',	'FLOAT', Beta),...
                makeAttributeProto('bias',	'FLOAT', K),...
                makeAttributeProto('size',	'INT',   WindowChannelSize)
                ];
inits          = [];
end