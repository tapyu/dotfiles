classdef ConverterForKerasGlobalAveragePooling2dLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a globalAveragePooling2dLayer into ONNX
    
    % Copyright 2018-2021 The MathWorks, Inc.
    
    methods
        function this = ConverterForKerasGlobalAveragePooling2dLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            
            % GAP node
            [gapName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            gapName                = makeUniqueName({nodeProto.name}, gapName);
            gapNode              = NodeProto;
            gapNode.op_type      = 'GlobalAveragePool';
            gapNode.name         = gapName;
            gapNode.input        = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            gapNode.output       = {gapName};
            
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
