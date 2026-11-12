classdef ConverterForONNXPreluLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a nnet.onnx.layer.PreluLayer into ONNX
    
    % Copyright 2018-2025 The MathWorks, Inc.
    
    methods
        function this = ConverterForONNXPreluLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            
            [NodeName, ~]       = legalizeNNTName(this, this.NNTLayer.Name);
            NodeName            = makeUniqueName({nodeProto.name}, NodeName);
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});
            SlopeName         	= [NodeName '_Slope'];
            [newNode, ~] = createNodeProto(this, 'SISO', 'PRelu', NodeName, ...
                        mapTensorNames(this, {this.InputLayerNames{1}, SlopeName}, TensorNameMap), {NodeName});
                                   
            % Make parameter Initializer for slope
            t1              = TensorProto;
            t1.name         = SlopeName;     
            t1.data_type	= TensorProto_DataType.FLOAT;
            slope           = this.NNTLayer.Alpha(:);
            t1.raw_data     = rawData(single(slope));
            
            % Set dimensions appropriate for Unidirectional broadcasting:            
            switch inputTensorLayout
                case {'snc', '1nc','nc'}
                    % Leave at length 1: [T N C] + [C] = [T N C], or
                    % [N C] + [C] = [N C]
                    t1.dims     = dimVector(numel(slope),1);    
                case {'snchw', 'nchw'}
                    % [(S) N C H W] + [C 1 1] = [(S) N C H W]
                    t1.dims     = dimVector(numel(slope),3);    
                case {'snchwd', 'nchwd'}
                    % [(S) N C H W D] + [C 1 1 1] = [(S) N C H W D]
                    t1.dims     = dimVector(numel(slope),4);   
                otherwise
                    error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', this.NNTLayer.Name));
            end
            
            nodeProto(end+1)        = newNode;
            parameterInitializers   = t1;
            % Make parameter Inputs
            networkInputs           = [];
            networkOutputs          = [];
                        
            % Update maps
            outputTensorName            = NodeName;
            TensorNameMap(this.NNTLayer.Name) = outputTensorName;
            TensorLayoutMap(outputTensorName) = inputTensorLayout;
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end

