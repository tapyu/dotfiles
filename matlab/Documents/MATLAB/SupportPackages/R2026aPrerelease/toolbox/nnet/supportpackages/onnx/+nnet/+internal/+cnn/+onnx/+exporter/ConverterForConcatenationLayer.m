classdef ConverterForConcatenationLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a nnet.cnn.layer.ConcatenationLayer into ONNX
    
    % Copyright 2019-2025 The MathWorks, Inc.
    
    methods
        function this = ConverterForConcatenationLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            % Generate Concat node
            import nnet.internal.cnn.onnx.*
            
            [onnxName, ~]       = legalizeNNTName(this, this.NNTLayer.Name);
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});
            dltDim = this.NNTLayer.Dim;            
            
            switch inputTensorLayout
                case 'nch'
                    onnxDims = [2 1];
                    concatAxis = onnxDims(dltDim);
                case 'nchw'
                    onnxDims = [2 3 1];
                    concatAxis = onnxDims(dltDim);
                case 'nchwd'
                    onnxDims = [2 3 4 1];
                    concatAxis = onnxDims(dltDim);
                case 'nc'
                    concatAxis = 1;
                case 'snc'
                    concatAxis = 2;
                case 'snch'
                    onnxDims = [3 2];
                    concatAxis = onnxDims(dltDim);
                case 'snchw'
                    onnxDims = [3 4 2];
                    concatAxis = onnxDims(dltDim);
                case 'snchwd'
                    onnxDims = [3 4 5 2];
                    concatAxis = onnxDims(dltDim);
                otherwise
                    error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', this.NNTLayer.Name));
            end
            
            % (1) Concatenate
            concatNodeName     = onnxName;
            concatNodeName     = makeUniqueName({nodeProto.name}, concatNodeName);
            [newNode, ~] = createNodeProto(this, 'Concat', concatNodeName,...
                inputTensorNames, {concatNodeName}, concatAxis);       
            
            nodeProto               = [nodeProto newNode];
            parameterInitializers   = [];
            networkInputs           = [];
            networkOutputs          = [];
            
            % Update maps
            outputTensorName = concatNodeName;
            TensorNameMap(this.NNTLayer.Name) = outputTensorName;
            TensorLayoutMap(outputTensorName) = inputTensorLayout;
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end

