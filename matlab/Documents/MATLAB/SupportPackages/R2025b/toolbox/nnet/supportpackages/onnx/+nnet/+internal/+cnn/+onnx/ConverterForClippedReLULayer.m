classdef ConverterForClippedReLULayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert an clippedReluLayer into ONNX
    
    % Copyright 2018-2021 The MathWorks, Inc.
    
    methods
        function this = ConverterForClippedReLULayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            % An NNT clippedReluLayer translates into a Relu followed by a Clip.
            import nnet.internal.cnn.onnx.*
            existingNodeNames = {nodeProto.name};
            [ONNXName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});
            % (1) Relu: Takes input X
            ReluNodeName            = [ONNXName '_Relu'];
            ReluNodeName            = makeUniqueName(existingNodeNames, ReluNodeName);
            newNodes(1)            = NodeProto;
            newNodes(1).op_type    = 'Relu';
            newNodes(1).name     	= ReluNodeName;
            newNodes(1).input      = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            newNodes(1).output    = {ReluNodeName};
            
            % (2) Clip: Takes input ReluNodeName
            clipNodeName        	= [ONNXName '_Clip'];
            clipNodeName            = makeUniqueName([existingNodeNames {ReluNodeName}], clipNodeName);
            clipNodeInput           = {ReluNodeName};
            clipNodeOutput          = {clipNodeName};
            clipMin                 = 0; 
            clipMax                 = this.NNTLayer.Ceiling;
            [newNodes(2), inits]    = createNodeProto(this, 'Clip', clipNodeName, clipNodeInput, clipNodeOutput, clipMin, clipMax);
            
            nodeProto               = [nodeProto newNodes];
            parameterInitializers   = inits;
            networkInputs           = [];
            networkOutputs          = [];

            % Update maps
            outputTensorName                  = clipNodeName;
            TensorNameMap(this.NNTLayer.Name) = outputTensorName;
            TensorLayoutMap(outputTensorName) = inputTensorLayout;
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end