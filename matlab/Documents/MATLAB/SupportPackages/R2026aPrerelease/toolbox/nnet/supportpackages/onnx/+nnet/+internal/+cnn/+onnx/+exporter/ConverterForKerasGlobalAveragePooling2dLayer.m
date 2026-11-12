classdef ConverterForKerasGlobalAveragePooling2dLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a globalAveragePooling2dLayer into ONNX
    
    % Copyright 2018-2025 The MathWorks, Inc.
    
    methods
        function this = ConverterForKerasGlobalAveragePooling2dLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            
            % GAP node
            [gapName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            gapName                = makeUniqueName({nodeProto.name}, gapName);
            gapNode = createNodeProto(this, 'SISO', 'GlobalAveragePool', gapName,...
                mapTensorNames(this, this.InputLayerNames(1), TensorNameMap), {gapName});
            
            nodeProto(end+1)        = gapNode;
            parameterInitializers   = [];
            networkInputs           = [];
            networkOutputs          = [];
            
            % Update maps
            TensorNameMap(this.NNTLayer.Name) = gapName;
            TensorLayoutMap(gapName)          = 'nchw';
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end
