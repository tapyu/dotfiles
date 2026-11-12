function [nodeProto, parameterInitializers] = createRescaleNodes(inputImageTensorName, opsetVersion, permIdx, layer, targetLowerBound, targetUpperBound)
    % This utility creates a series of nodes and initializers that will
    % perform the following transformation to the input: 
    % Y = targetLowerBound + ((targetUpperBound - targetLowerBound) / (MaxVal -
    % MinVal)) .* (X - MinVal)

    %   Copyright 2019 The MathWorks, Inc.

    import nnet.internal.cnn.onnx.*
    MinVal = layer.Min; 
    MaxVal = layer.Max; 
    scaleFactor = single(targetUpperBound - targetLowerBound) ./ single(MaxVal - MinVal);
    shiftValue  = targetLowerBound - (scaleFactor .* MinVal);
    
    % Scale the centered input by the scaling factor.
    mulNodeName             = [inputImageTensorName '_Mul'];
    scalingName             = [inputImageTensorName '_Scaling'];
    mulNode                 = NodeProto; 
    mulNode.op_type         = 'Mul';
    mulNode.name            = mulNodeName;
    mulNode.input           = {inputImageTensorName, scalingName};
    mulNode.output          = {mulNodeName};
    if opsetVersion < 7
        mulNode.attribute = [... 
            makeAttributeProto('broadcast', 'INT', 1),...
            makeAttributeProto('axis', 'INT', 1)];
    end

    % Scale node parameter initializer
    mulTensor               = TensorProto;
    mulTensor.name          = scalingName;
    mulTensor.data_type     = TensorProto_DataType.FLOAT;
    scaleFactor             = permute(scaleFactor, permIdx); % NNT is hwc/hwdc, onnx is chw/chwd (with n=1 and broadcast)
    mulTensor.raw_data      = rawData(single(scaleFactor)); 
    mulTensor.dims          = dimVector(size(scaleFactor), numel(permIdx)); % dims = NCHW or NCHWD 
    if opsetVersion < 7
        mulTensor.dims      = mulTensor.dims(2:end); % dims = CHW or CHWD
    end

    % Shift back the input to start from the target min
    addNodeName             = [inputImageTensorName '_Add'];
    shiftName               = [inputImageTensorName '_Shift'];
    addNode                 = NodeProto;
    addNode.op_type         = 'Add';
    addNode.name            = addNodeName;
    addNode.input           = {mulNodeName, shiftName};
    addNode.output          = {addNodeName};
    if opsetVersion < 7
        addNode.attribute = [...
            makeAttributeProto('broadcast', 'INT', 1),...
            makeAttributeProto('axis', 'INT', 1)];
    end
    
    % Target min parameter initializer
    addTensor               = TensorProto;
    addTensor.name          = shiftName;
    addTensor.data_type     = TensorProto_DataType.FLOAT;
    shiftValue              = permute(shiftValue, permIdx); % NNT is hwc/hwdc, onnx is chw/chwd (with n=1 and broadcast)
    addTensor.raw_data      = rawData(single(shiftValue));
    addTensor.dims          = dimVector(size(shiftValue), numel(permIdx)); % dims = NCHW or NCHWD 
    if opsetVersion < 7
        addTensor.dims      = addTensor.dims(2:end); % dims = CHW or CHWD
    end
    
    % Finish up
    nodeProto               = [mulNode addNode];
    parameterInitializers   = [mulTensor addTensor];    
end

