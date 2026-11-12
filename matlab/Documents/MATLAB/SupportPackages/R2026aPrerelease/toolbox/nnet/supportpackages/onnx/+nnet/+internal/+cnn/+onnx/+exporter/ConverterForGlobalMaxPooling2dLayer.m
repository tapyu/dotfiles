classdef ConverterForGlobalMaxPooling2dLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a builtin globalMaxPooling2DLayer into ONNX
        
    % Copyright 2019-2025 The MathWorks, Inc.

    methods
        function this = ConverterForGlobalMaxPooling2dLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            
            % GAP node
            [gmpName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            gmpName                = makeUniqueName({nodeProto.name}, gmpName);
            gapNode = createNodeProto(this, 'SISO', 'GlobalMaxPool', gmpName,...
                    mapTensorNames(this, this.InputLayerNames(1), TensorNameMap), {gmpName});
            
            nodeProto(end+1)        = gapNode;
            parameterInitializers   = [];
            networkInputs           = [];
            networkOutputs          = [];
            
            % Update maps
            TensorNameMap(this.NNTLayer.Name) = gmpName;
            TensorLayoutMap(gmpName)          = 'nchw';
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end
