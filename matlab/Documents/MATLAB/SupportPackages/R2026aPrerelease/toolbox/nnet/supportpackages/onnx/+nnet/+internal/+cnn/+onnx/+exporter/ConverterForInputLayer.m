classdef ConverterForInputLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a InputLayer into ONNX
        
    % Copyright 2023-2025 The MathWorks, Inc.

    methods
        function this = ConverterForInputLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            inputTensorName        = legalizeNNTName(this, this.NNTLayer.Name);
            inputTensorName        = makeUniqueName({nodeProto.name}, inputTensorName);
            nntLayer               = this.NNTLayer; 
            DLTShape               = nntLayer.InputSize;
            DLTFormat              = nntLayer.InputFormat;

            % Error if output formats has a U dim
            % NB: U dims unsupported because it is unclear what ONNX
            % formats these should correspond to.
            if contains(DLTFormat, 'U')
                error(message('nnet_cnn_onnx:onnx:UnsupportedUnspecifiedFormats', nntLayer.Name));
            end

            % Error if output format does not have a C dim
            if ~contains(DLTFormat, 'C')
                error(message('nnet_cnn_onnx:onnx:UnsupportedNoCDim', nntLayer.Name));
            end

            % Remove B dimension and corresponding size to be added later
            if contains(DLTFormat, 'B')
                batchDim = find(DLTFormat=='B');
                DLTShape(batchDim) = [];
                DLTFormat(batchDim) = [];
            end

            % Remove C dimension and corresponding size to be added later
            channelDim = find(DLTFormat=='C');
            channelSize = DLTShape(channelDim);
            DLTShape(channelDim) = [];
            DLTFormat(channelDim) = [];

            % Remove T dimension and corresponding size to be added later
            % (if needed)
            hasTdimension = contains(DLTFormat, 'T');
            if hasTdimension
                timeDim = find(DLTFormat=='T');
                DLTShape(timeDim) = [];
                DLTFormat(timeDim) = [];
            end

            % Convert to ONNX format ordering
            % NB: From the list of possible formats, ONNX only supports the
            % following:
            % Spatial:          nch, nchw, nchwd
            % Temporal:         snc
            % Feature:          nc
            % Spatio-temporal:  snch, snchw, snchwd
            % NB: DLTFormat is now some combination of an arbitrary number
            % of U and S dims. However, only char.empty, S, SS, SSS are
            % valid, and only S, SS, SSS require transformation.
            switch DLTFormat
                case 'S'
                    outputTensorLayout = 'h';
                case 'SS'
                    outputTensorLayout = 'hw';
                case 'SSS'
                    outputTensorLayout = 'hwd';
                otherwise
                    % Let DLTFormat pass through unchanged
                    % NB: There are only two options here:
                    %       - Empty char    -> valid
                    %       - Nonempty char -> invalid (will error later)
                    outputTensorLayout = DLTFormat;
            end
            ONNXShape = num2cell(DLTShape);
            
            % Add the channel dimension
            outputTensorLayout = append('c', outputTensorLayout);
            ONNXShape = [{channelSize}, ONNXShape];

            % Add the batch dimension
            outputTensorLayout = append('n', outputTensorLayout);
            ONNXShape = [{this.BatchSizeToExport}, ONNXShape];

            % Add the time dimension (if needed)
            if hasTdimension
                sequenceDimensionName = [inputTensorName '_SequenceLength']; 
                outputTensorLayout = append('s', outputTensorLayout);
                ONNXShape = [{sequenceDimensionName}, ONNXShape];
            end

            % Make a ValueInfoProto describing the input image tensor.
            networkInputs      	 = makeValueInfoProtoFromDimensions(inputTensorName, TensorProto_DataType.FLOAT, ONNXShape);
            networkOutputs     	 = [];
            
            % Generate a ValueInfoProto describing the input tensor.  
            curNodeProto = []; 
            parameterInitializers = [];

            TensorNameMap(this.NNTLayer.Name) = inputTensorName;
            TensorLayoutMap(inputTensorName) = outputTensorLayout;
            nodeProto = [nodeProto curNodeProto]; 
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end
