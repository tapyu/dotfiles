function [InstanceNormalizationNode, inits] = createInstanceNormalizationNode(~, name, input, output, epsilon)
% A helper function to create a InstanceNormalization operator of the specified opset
% version. 

%   Copyright 2025 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
InstanceNormalizationNode = NodeProto;
InstanceNormalizationNode.op_type   = 'InstanceNormalization';
InstanceNormalizationNode.name      = name;
InstanceNormalizationNode.input     = input;
InstanceNormalizationNode.output    = output;
InstanceNormalizationNode.attribute = makeAttributeProto('epsilon', 'FLOAT', epsilon);

inits               = [];
end