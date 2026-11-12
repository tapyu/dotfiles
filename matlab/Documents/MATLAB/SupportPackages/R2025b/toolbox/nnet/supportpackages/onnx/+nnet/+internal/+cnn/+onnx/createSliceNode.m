function [sliceNode, inits] = createSliceNode(opset, name, input, output, starts, ends)
% A helper function to create a Slice operator of the specified opset
% version. 

%   Copyright 2021-2024 The MathWorks, Inc.

import nnet.internal.cnn.onnx.*
sliceNode = NodeProto;
sliceNode.op_type   = 'Slice';
sliceNode.name      = name;
sliceNode.input     = input;
sliceNode.output    = output;
inits               = [];
if opset < 10
    sliceNode.attribute = [...
        makeAttributeProto('starts', 'INTS', starts), ...
        makeAttributeProto('ends',   'INTS', ends)];    
else
    % In Opset 10 and later, 'starts' and 'ends' are node inputs. Create
    % initializers for them.
    if ischar(starts)
        sliceNode.input     = [sliceNode.input {starts ends}];
    else
        startsInit              = TensorProto;
        startsInit.name         = [name '_starts'];
        startsInit.data_type    = TensorProto_DataType.INT64;
        startsInit.raw_data     = rawData(int64(starts));
        startsInit.dims         = dimVector(numel(starts), 1);
    
        endsInit            = TensorProto;
        endsInit.name       = [name '_ends'];
        endsInit.data_type  = TensorProto_DataType.INT64;
        endsInit.raw_data   = rawData(int64(ends));
        endsInit.dims       = dimVector(numel(ends), 1);
    
        sliceNode.input     = [sliceNode.input {startsInit.name endsInit.name}];
        inits = [startsInit, endsInit];
    end
end
end