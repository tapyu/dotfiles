classdef ConverterForSpaceToDepthLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a spaceToDepthLayer into ONNX.

    % Copyright 2020-2021 The MathWorks, Inc.
    
    methods
        function this = ConverterForSpaceToDepthLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            
            inputTensorNames = mapTensorNames(this, this.InputLayerNames(:)', TensorNameMap);
            
            if(this.NNTLayer.BlockSize(1)~=this.NNTLayer.BlockSize(2))
				error(message('nnet_cnn_onnx:onnx:spaceToDepthRectangular', this.NNTLayer.BlockSize(1), this.NNTLayer.BlockSize(2)));
            else
                [ONNXName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
                ONNXName                = makeUniqueName({nodeProto.name}, ONNXName);
                newNode                 = NodeProto;
                newNode.op_type         = 'SpaceToDepth';
                newNode.name            = ONNXName;
                newNode.input           = inputTensorNames;
                newNode.output{1}       = ONNXName;
                newNode.attribute       = makeAttributeProto('blocksize', 'INT', this.NNTLayer.BlockSize(1));
            
                nodeProto(end+1)        = newNode;
                parameterInitializers   = [];
                networkInputs           = [];
                networkOutputs          = [];
            end  
            
             % Update maps
            TensorNameMap(this.NNTLayer.Name) = ONNXName;
            TensorLayoutMap(ONNXName)         = 'nchw';

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end
