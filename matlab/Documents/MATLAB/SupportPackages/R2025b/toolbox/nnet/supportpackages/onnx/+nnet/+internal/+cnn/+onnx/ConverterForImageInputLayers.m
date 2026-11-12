classdef ConverterForImageInputLayers < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert an ImageInputLayer or Image3dInputLayer into ONNX.
    % Normalization is considered
    
    % Copyright 2018-2022 The MathWorks, Inc.
    
    methods
        function this = ConverterForImageInputLayers(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            % The layer can be either an ImageInputLayer or
            % Image3dInputLayer. Determine how to permute the average image
            % and what the tensor output will be based on the image shape
            inputImageTensorName    = legalizeNNTName(this, this.NNTLayer.Name);
            inputImageTensorName    = makeUniqueName({nodeProto.name}, inputImageTensorName);
            DLTShape                = this.OutputSize{1};                               % HWC or HWCD            
            nntLayer = this.NNTLayer;
            if isa(nntLayer, 'nnet.cnn.layer.ImageInputLayer')
                permIdx             = [4 3 1 2];
                ONNXShape           = [{this.BatchSizeToExport}, num2cell(DLTShape([3 1 2]))];       % {NCHW}
                outputTensorLayout  = 'nchw';

                if nntLayer.SplitComplexInputs % SplitComplexInputs is not supported 
                    error(message('nnet_cnn_onnx:onnx:UnsupportedSplitComplexInputs', nntLayer.Name));
                end
            elseif isa(nntLayer, 'nnet.cnn.layer.Image3DInputLayer')
                permIdx             = [5 4 1 2 3];
                ONNXShape           = [{this.BatchSizeToExport}, num2cell(DLTShape([4 1 2 3]))];     % {NCHWD}
                outputTensorLayout  = 'nchwd';                
            else
                error(message("nnet_cnn_onnx:onnx:UnexpectedLayer", 'nnet.cnn.layer.ImageInputLayer', 'nnet.cnn.layer.Image3DInputLayer', class(this.NNTLayer)));
            end
            
            % Generate a ValueInfoProto describing the input image tensor
            % and optionally generate additional nodes to perform normalization.
            switch char(nntLayer.Normalization) % cast to char in the case of function handle
                case 'zerocenter'
                    [curNodeProto, parameterInitializers] = createZeroCenterNodes(inputImageTensorName, this.OpsetVersion, permIdx, nntLayer);
                case 'zscore'
                    [curNodeProto, parameterInitializers] = createZScoreNodes(inputImageTensorName,  this.OpsetVersion, permIdx, nntLayer);
                case 'rescale-symmetric'
                    [curNodeProto, parameterInitializers] = createRescaleNodes(inputImageTensorName, this.OpsetVersion, permIdx, nntLayer, -1, 1);
                case 'rescale-zero-one'
                    [curNodeProto, parameterInitializers] = createRescaleNodes(inputImageTensorName, this.OpsetVersion, permIdx, nntLayer,  0, 1);
                case 'none'
                    curNodeProto = []; 
                    parameterInitializers = [];
                otherwise
                    error(message('nnet_cnn_onnx:onnx:NormalizationUnsupportedForExport'));
            end
            
            % Make a ValueInfoProto describing the input image tensor.
            networkInputs      	 = makeValueInfoProtoFromDimensions(inputImageTensorName, TensorProto_DataType.FLOAT, ONNXShape);
            networkOutputs     	 = [];
            
            % Update maps
            if ~isempty(parameterInitializers)
                outputTensorName = curNodeProto(end).name;
            else
                outputTensorName = inputImageTensorName;
            end
            
            TensorNameMap(this.NNTLayer.Name) = outputTensorName;
            TensorLayoutMap(outputTensorName) = outputTensorLayout;
            nodeProto = [nodeProto curNodeProto]; 
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end
