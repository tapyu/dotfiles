function [nodeProto, parameterInitializers] = createZeroCenterNodes(inputImageTensorName, opsetVersion, permIdx, layer)
    % This utility creates a series of nodes and initializers that will
    % perform centering on the input given a known mean: 
    % Y = X - Mean 

    %   Copyright 2019 The MathWorks, Inc.

    import nnet.internal.cnn.onnx.*
    meanImage = layer.Mean;
    
    % Sub node to subtract the Mean
    subNodeName           = [inputImageTensorName '_Sub'];
    avgImgName            = [inputImageTensorName '_Mean'];
    subNode               = NodeProto;
    subNode.op_type       = 'Sub';
    subNode.name          = subNodeName;
    subNode.input         = {inputImageTensorName, avgImgName};
    subNode.output        = {subNode.name};
    if opsetVersion < 7
        subNode.attribute = [...
            makeAttributeProto('broadcast', 'INT', 1),...
            makeAttributeProto('axis', 'INT', 1)];
    end

    % Sub node parameter initializer
    subTensor             = TensorProto;
    subTensor.name        = avgImgName;
    subTensor.data_type   = TensorProto_DataType.FLOAT;
    permutedImg           = permute(meanImage, permIdx); % NNT is hwc/hwdc, onnx is chw/chwd (with n=1 and broadcast)
    subTensor.raw_data    = rawData(single(permutedImg));
    subTensor.dims        = dimVector(size(permutedImg),numel(permIdx)); % dims = NCHW or NCHWD 
    if opsetVersion < 7
        subTensor.dims    = subTensor.dims(2:end); % dims = CHW or CHWD
    end

    % Finish up 
    parameterInitializers = subTensor;
    nodeProto             = subNode;
end
