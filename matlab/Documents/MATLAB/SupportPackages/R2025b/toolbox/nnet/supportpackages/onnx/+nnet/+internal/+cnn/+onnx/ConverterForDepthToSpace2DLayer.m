classdef ConverterForDepthToSpace2DLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a depthToSpace2dLayer into ONNX.

    % Copyright 2020-2021 The MathWorks, Inc.
    
    methods
        function this = ConverterForDepthToSpace2DLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            
            inputTensorNames = mapTensorNames(this, this.InputLayerNames(:)', TensorNameMap);
            
            [ONNXName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            ONNXName                = makeUniqueName({nodeProto.name}, ONNXName);
            newNode                 = NodeProto;
            newNode.op_type         = 'DepthToSpace';
            newNode.name            = ONNXName;
            newNode.input           = inputTensorNames;
            newNode.output{1}       = ONNXName;            
            
            if(this.NNTLayer.BlockSize(1)~=this.NNTLayer.BlockSize(2))
				error(message('nnet_cnn_onnx:onnx:depthToSpaceRectangular'));
                
            elseif this.OpsetVersion < 11    
                newNode.attribute       = [...
                    makeAttributeProto('blocksize', 'INT', this.NNTLayer.BlockSize(1)),...
                    ];
            
                nodeProto(end+1)        = newNode;
                parameterInitializers   = [];
                networkInputs           = [];
                networkOutputs          = [];
            else
                newNode.attribute       = [...
                    makeAttributeProto('blocksize', 'INT', this.NNTLayer.BlockSize(1)),...
                    makeAttributeProto('mode', 'STRING', char(this.NNTLayer.Mode))
                    ];
            
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
