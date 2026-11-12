classdef ConverterForIdentityLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert an IdentityLayer into ONNX
    
    % Copyright 2024 The MathWorks, Inc.
    
    methods
        function this = ConverterForIdentityLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
            inputTensorNames        = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout       = TensorLayoutMap(inputTensorNames{1});
            
            % Make the nodeProto
            newNode           = NodeProto;
            newNode.op_type	= 'Identity';
            newNode.name      = onnxName;
            newNode.input     = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            newNode.output    = {onnxName};
            
            nodeProto(end+1)        = newNode;
            parameterInitializers   = [];
            networkInputs           = [];
            networkOutputs          = [];
            
            % Update maps
            TensorNameMap(this.NNTLayer.Name) = onnxName;
            TensorLayoutMap(onnxName) = inputTensorLayout;
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end

