classdef ConverterForFeatureInputLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a FeatureInputLayer into ONNX.
    % Normalization is considered
    
    % Copyright 2020-2025 The MathWorks, Inc.
    
    methods
        function this = ConverterForFeatureInputLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            % The layer can only ever receice feature data, in the form
            % numObservations x numFeatures
            featureInputTensorName    = legalizeNNTName(this, this.NNTLayer.Name);
            featureInputTensorName    = makeUniqueName({nodeProto.name}, featureInputTensorName);
            DLTShape                = this.OutputSize{1};                               % C            
            nntLayer = this.NNTLayer;
            
            if numel(DLTShape) == 1 % C
                % Mean is [1 C]
                perm = [1 2];
                outputTensorLayout = 'nc';
                outputTensorSize = [{this.BatchSizeToExport} num2cell(DLTShape)];
            else 
                error(message('nnet_cnn_onnx:onnx:UnexpectedDLTInputSize', nntLayer.Name, numel(DLTShape)));
            end

            if nntLayer.SplitComplexInputs % SplitComplexInputs is not supported 
                error(message('nnet_cnn_onnx:onnx:UnsupportedSplitComplexInputs', nntLayer.Name));
            end
            
            % Generate a ValueInfoProto describing the input tensor and
            % optionally generate additional nodes to perform normalization.  
            switch char(nntLayer.Normalization) % cast to char in case of function handle as normalization. 
                case 'zerocenter'
                    [curNodeProto, parameterInitializers] = createNodeProto(this, 'ZeroCenter', featureInputTensorName, perm, nntLayer);
                case 'zscore'
                    [curNodeProto, parameterInitializers] = createNodeProto(this, 'ZScore', featureInputTensorName, perm, nntLayer);
                case 'rescale-symmetric'
                    [curNodeProto, parameterInitializers] = createNodeProto(this, 'Rescale', featureInputTensorName, perm, nntLayer, -1, 1);
                case 'rescale-zero-one'
                    [curNodeProto, parameterInitializers] = createNodeProto(this, 'Rescale', featureInputTensorName, perm, nntLayer,  0, 1);
                case 'none'
                    curNodeProto = []; 
                    parameterInitializers = [];
                otherwise
                    error(message('nnet_cnn_onnx:onnx:NormalizationUnsupportedForExport'));
            end
                        
            % Make a ValueInfoProto describing the input image tensor.
            networkInputs      	 = makeValueInfoProtoFromDimensions(featureInputTensorName, TensorProto_DataType.FLOAT, outputTensorSize);
            networkOutputs     	 = [];
            
            % Update maps
            if ~isempty(parameterInitializers)
                outputTensorName = curNodeProto(end).name;
            else
                outputTensorName = featureInputTensorName;
            end
            
            TensorNameMap(this.NNTLayer.Name) = outputTensorName;
            TensorLayoutMap(outputTensorName) = outputTensorLayout;
            nodeProto = [nodeProto curNodeProto]; 
        end
    end
end
