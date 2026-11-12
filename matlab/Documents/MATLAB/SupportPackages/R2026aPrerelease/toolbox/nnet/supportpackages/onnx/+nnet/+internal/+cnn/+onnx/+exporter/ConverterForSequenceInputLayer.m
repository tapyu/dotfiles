classdef ConverterForSequenceInputLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a SequenceInputLayer into ONNX
        
    % Copyright 2018-2025 The MathWorks, Inc.

    methods
        function this = ConverterForSequenceInputLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            
            % The layer can input vector sequences, 2d image sequences, or
            % 3d image sequences. Determine how to permute the mean
            % and what the tensor output will be based on the type of
            % sequence input

            seqInputTensorName        = legalizeNNTName(this, this.NNTLayer.Name);
            seqInputTensorName        = makeUniqueName({nodeProto.name}, seqInputTensorName);
            DLTShape                  = this.OutputSize{1}; % C, HC, HWC, or HWDC    
            nntLayer                  = this.NNTLayer; 
            sequenceDimensionName     = [seqInputTensorName '_SequenceLength']; 
            
            if numel(DLTShape) == 1 % C
                % Mean is [C 1]
                perm = [2 1];
                numParamDims = 3; % 'snc'
                outputTensorLayout = 'snc';
                outputTensorSize = [{sequenceDimensionName} {this.BatchSizeToExport} num2cell(DLTShape)];
             elseif numel(DLTShape) == 2 % HC
                % Mean is [H C]
                perm = [2 1];
                numParamDims = 4; % 'snch'
                outputTensorLayout = 'snch';
                outputTensorSize = [{sequenceDimensionName} {this.BatchSizeToExport} num2cell(DLTShape(perm))];
            elseif numel(DLTShape) == 3 % HWC
                % Mean is [H W C]
                perm = [3 1 2];
                numParamDims = 5; % 'snchw'
                outputTensorLayout = 'snchw';
                outputTensorSize = [{sequenceDimensionName} {this.BatchSizeToExport} num2cell(DLTShape(perm))];
            else % HWDC
                % Mean is [H W D C]
                assert(numel(DLTShape) == 4,...
                    message('nnet_cnn_onnx:onnx:UnexpectedDLTInputSize', this.NNTLayer.Name, numel(DLTShape)+1));
                perm = [4 1 2 3];
                numParamDims = 6; % 'snchwd'
                outputTensorLayout = 'snchwd';
                outputTensorSize = [{sequenceDimensionName} {this.BatchSizeToExport} num2cell(DLTShape(perm))];
            end

            if nntLayer.SplitComplexInputs % SplitComplexInputs is not supported 
                error(message('nnet_cnn_onnx:onnx:UnsupportedSplitComplexInputs', nntLayer.Name));
            end
            
            
            % Generate a ValueInfoProto describing the input image tensor
            % and optionally generate additional nodes to perform normalization.  
            switch char(nntLayer.Normalization) % cast to char in case of function handle as normalization. 
                case 'zerocenter'
                    [curNodeProto, parameterInitializers] = createNodeProto(this, 'ZeroCenter', seqInputTensorName, perm, nntLayer);
                case 'zscore'
                    [curNodeProto, parameterInitializers] = createNodeProto(this, 'ZScore', seqInputTensorName, perm, nntLayer);
                case 'rescale-symmetric'
                    [curNodeProto, parameterInitializers] = createNodeProto(this, 'Rescale', seqInputTensorName, perm, nntLayer, -1, 1);
                case 'rescale-zero-one'
                    [curNodeProto, parameterInitializers] = createNodeProto(this, 'Rescale', seqInputTensorName, perm, nntLayer,  0, 1);
                case 'none'
                    curNodeProto = []; 
                    parameterInitializers = [];
                otherwise
                    error(message('nnet_cnn_onnx:onnx:NormalizationUnsupportedForExport'));
            end
            
            for i = 1:numel(parameterInitializers)
                parameterInitializers(i).dims = prependOnes(parameterInitializers(i).dims, numParamDims); 
            end
                                    
            networkInputs           = makeValueInfoProtoFromDimensions(...
                seqInputTensorName, ...
                TensorProto_DataType.FLOAT, ...
                outputTensorSize);
            networkOutputs          = [];
            
            % Update maps
            if ~isempty(parameterInitializers)
                outputTensorName = curNodeProto(end).name;
            else
                outputTensorName = seqInputTensorName; 
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
