classdef ConverterForONNXFlattenInto2dLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a nnet.onnx.layer.FlattenInto2dLayer into ONNX
    
    % Copyright 2021-2025 The MathWorks, Inc.
    
    methods
        function this = ConverterForONNXFlattenInto2dLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            % Generate Flatten operator
            import nnet.internal.cnn.onnx.*
            
            [onnxName, ~] = legalizeNNTName(this, this.NNTLayer.Name);
            existingNodeNames   = {nodeProto.name};
            
            % (1) Flatten
            flattenNodeName         = onnxName;
            flattenNodeName         = makeUniqueName(existingNodeNames, flattenNodeName);
            flattenNode             = createNodeProto(this, 'Flatten', flattenNodeName,...
                mapTensorNames(this, this.InputLayerNames(1), TensorNameMap),{flattenNodeName},1);

            nodeProto               = [nodeProto flattenNode];
            parameterInitializers   = [];
            networkInputs           = [];
            networkOutputs          = [];            
            
            % Update maps
            outputTensorName                  = flattenNodeName;
            TensorNameMap(this.NNTLayer.Name) = outputTensorName;
            TensorLayoutMap(outputTensorName) = 'nc';
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end

