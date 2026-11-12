classdef ConverterForClippedReLULayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert an clippedReluLayer into ONNX
    
    % Copyright 2018-2025 The MathWorks, Inc.
    
    methods
        function this = ConverterForClippedReLULayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
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
            [reluNode, reluInits] = createNodeProto(this, 'SISO','Relu', ReluNodeName, ...
                mapTensorNames(this, this.InputLayerNames(1), TensorNameMap), {ReluNodeName});
            
            % (2) Clip: Takes input ReluNodeName
            clipNodeName        	= [ONNXName '_Clip'];
            clipNodeName            = makeUniqueName([existingNodeNames {ReluNodeName}], clipNodeName);
            clipMin                 = 0; 
            clipMax                 = this.NNTLayer.Ceiling;
            [clipNode, clipInits]    = createNodeProto(this, 'Clip', clipNodeName, {ReluNodeName}, {clipNodeName}, clipMin, clipMax);
            
            nodeProto               = [nodeProto reluNode clipNode];
            parameterInitializers   = [reluInits clipInits];
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