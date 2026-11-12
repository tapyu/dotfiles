function [BatchnormNode, inits] = createBatchNormalizationNode(opset, name, input, output, epsilon, Scale, Offset, TrainedMean, TrainedVariance)
% A helper function to create a BatchNormalization operator of the specified opset
% version. 

%   Copyright 2025 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
BatchnormNode = NodeProto;
BatchnormNode.op_type   = 'BatchNormalization';
BatchnormNode.name      = name;
BatchnormNode.input     = input;
BatchnormNode.output    = output;
BatchnormNode.attribute = makeAttributeProto('epsilon', 'FLOAT', epsilon);
if opset < 7
    BatchnormNode.attribute(1,2) = makeAttributeProto('is_test', 'INT', 1);
end
% Make parameter Initializers for: scale, B, mean, var
t1 = TensorProto;
t1.name = [name '_scale'];
t1.data_type = TensorProto_DataType.FLOAT;
t1.raw_data = rawData(single(Scale));
t1.dims = dimVector(numel(Scale),1);           % NNT data: numChannels -1 (1D input) or 1-1-numChannels (2D input) or 1-1-1-numChannels (3D input)

t2 = TensorProto;
t2.name = [name '_B'];
t2.data_type = TensorProto_DataType.FLOAT;
t2.raw_data = rawData(single(Offset));
t2.dims = dimVector(numel(Offset),1);          % NNT data: numChannels -1 (1D input) or 1-1-numChannels (2D input) or 1-1-1-numChannels (3D input)

t3 = TensorProto;
t3.name = [name '_mean'];
t3.data_type = TensorProto_DataType.FLOAT;
t3.raw_data = rawData(single(TrainedMean));
t3.dims = dimVector(numel(TrainedMean),1);  	% NNT data: numChannels -1 (1D input) or 1-1-numChannels (2D input) or 1-1-1-numChannels (3D input)

t4 = TensorProto;
t4.name = [name '_var'];
t4.data_type = TensorProto_DataType.FLOAT;
t4.raw_data = rawData(single(TrainedVariance));
t4.dims = dimVector(numel(TrainedVariance),1);	% NNT data: numChannels -1 (1D input) or 1-1-numChannels (2D input) or 1-1-1-numChannels (3D input)
inits               = [t1 t2 t3 t4];
end