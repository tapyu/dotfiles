classdef ConverterForPointCloudInputLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a PointCloudInputLayer into ONNX.
    % Normalization is considered
    
    % Copyright 2022 The MathWorks, Inc.
    
    methods
        function this = ConverterForPointCloudInputLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            % Determine how to permute the average image
            % and what the tensor output will be based on the image shape
            inputPointCloudTensorName    = legalizeNNTName(this, this.NNTLayer.Name);
            inputPointCloudTensorName    = makeUniqueName({nodeProto.name}, inputPointCloudTensorName);
            DLTShape                = this.OutputSize{1};                               % HC or HWC
            nntLayer = this.NNTLayer;

            if numel(DLTShape) == 2 % Unorganized Point Cloud Input
                permIdx             = [3 2 1];  % SCB/HCN -> NCH
                ONNXShape           = [{this.BatchSizeToExport}, num2cell(DLTShape([2 1]))];       % {NCH}
                outputTensorLayout  = 'nch';
     
            else % numel(DLTShape) == 3. Organized Point Cloud Input
                permIdx             = [4 3 1 2];    %SSCB/HWCN -> NCHW
                ONNXShape           = [{this.BatchSizeToExport}, num2cell(DLTShape([3 1 2]))];       % {NCHW}
                outputTensorLayout  = 'nchw';
            end
            
            % Don't need to check for SplitComplexInputs (nnt.SplitComplexInputs) as the
            % pointCloudInputLayer itself doesn't support it.
            
            % Generate a ValueInfoProto describing the input point cloud tensor
            % and optionally generate additional nodes to perform normalization.
            switch char(nntLayer.Normalization) % cast to char in the case of function handle
                case 'zerocenter'
                    [curNodeProto, parameterInitializers] = createZeroCenterNodes(inputPointCloudTensorName, this.OpsetVersion, permIdx, nntLayer);
                case 'zscore'
                    [curNodeProto, parameterInitializers] = createZScoreNodes(inputPointCloudTensorName,  this.OpsetVersion, permIdx, nntLayer);
                case 'rescale-symmetric'
                    [curNodeProto, parameterInitializers] = createRescaleNodes(inputPointCloudTensorName, this.OpsetVersion, permIdx, nntLayer, -1, 1);
                case 'rescale-zero-one'
                    [curNodeProto, parameterInitializers] = createRescaleNodes(inputPointCloudTensorName, this.OpsetVersion, permIdx, nntLayer,  0, 1);
                case 'none'
                    curNodeProto = []; 
                    parameterInitializers = [];
                otherwise
                    error(message('nnet_cnn_onnx:onnx:NormalizationUnsupportedForExport'));
            end
            
            % Make a ValueInfoProto describing the input image tensor.
            networkInputs      	 = makeValueInfoProtoFromDimensions(inputPointCloudTensorName, TensorProto_DataType.FLOAT, ONNXShape);
            networkOutputs     	 = [];
            
            % Update maps
            if ~isempty(parameterInitializers)
                outputTensorName = curNodeProto(end).name;
            else
                outputTensorName = inputPointCloudTensorName;
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
