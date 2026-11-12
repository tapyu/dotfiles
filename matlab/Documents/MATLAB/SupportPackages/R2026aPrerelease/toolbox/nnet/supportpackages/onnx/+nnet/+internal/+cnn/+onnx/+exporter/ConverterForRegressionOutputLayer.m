classdef ConverterForRegressionOutputLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a RegressionOutputLayer into ONNX
    
    % Copyright 2018-2025 The MathWorks, Inc.
    
    methods
        function this = ConverterForRegressionOutputLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            inputTensorNames   = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            inputTensorLayout = TensorLayoutMap(inputTensorNames{1});
            dltOutputSize     = this.InputLayerSizes{1};                            % dltOutputSize is either [1 1 C] or [C]
            
            switch inputTensorLayout
                case {'nchw', 'nchwd'}
                    if DLTTensorIsFlattened(dltOutputSize)
                        % Output a Flatten(1) operator
                        flattenNodeName        = [inputTensorNames{1} '_Flatten'];
                        flattenNodeName        = makeUniqueName({nodeProto.name}, flattenNodeName);
                        [FlattenNode, ~] = createNodeProto(this, 'Flatten', flattenNodeName, ...
                            inputTensorNames(1), {flattenNodeName}, 1);
                        outputTensorName       = flattenNodeName;
                        outputTensorSize       = [{this.BatchSizeToExport}, num2cell(dltOutputSize(end))];	% {N C}
                        newNode                = FlattenNode;
                    else
                        newNode                = [];
                        outputTensorName       = inputTensorNames{1};
                        outputTensorSize       = [{this.BatchSizeToExport}, num2cell(dltOutputSize([end 1:(end-1)]))];	% {N C H W} or {N C H W D}
                    end
                case {'snc', 'snchw', 'snchwd'}
                    outputTensorName        = inputTensorNames{1};
                    outputTensorSize        = [{1 this.BatchSizeToExport}, num2cell(dltOutputSize([end 1:(end-1)]))];	% {C}, {1 N C H W}, or {1 N C H W D}
                    newNode                 = [];
                case 'nc'
                    assert(isscalar(dltOutputSize),...
                        message('nnet_cnn_onnx:onnx:UnexpectedDLTInputSize', this.NNTLayer.Name, numel(dltOutputSize)+1));
                    outputTensorName        = inputTensorNames{1};
                    outputTensorSize       = [{this.BatchSizeToExport}, num2cell(dltOutputSize)];
                    newNode = [];
                case '1nc'
                    % seq2last
                    assert(isscalar(dltOutputSize),...
                        message('nnet_cnn_onnx:onnx:UnexpectedDLTInputSize', this.NNTLayer.Name, numel(dltOutputSize)+1));
                    % Output a Flatten(2) operator
                    flattenNodeName        = [inputTensorNames{1} '_Flatten'];
                    flattenNodeName        = makeUniqueName({nodeProto.name}, flattenNodeName);
                    [FlattenNode, ~] = createNodeProto(this, 'Flatten', flattenNodeName, ...
                            inputTensorNames(1), {flattenNodeName}, 2);
                    outputTensorName       = flattenNodeName;
                    outputTensorSize       = num2cell([this.BatchSizeToExport dltOutputSize]);	% [N C] for seq2last RNNs.
                    newNode                = FlattenNode;
                otherwise
                    error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', this.NNTLayer.Name));
            end
            
            nodeProto               = [nodeProto newNode];
            parameterInitializers   = [];
            networkInputs           = [];
            networkOutputs          = makeValueInfoProtoFromDimensions(outputTensorName, TensorProto_DataType.FLOAT, outputTensorSize);
        end
    end
end

function tf = DLTTensorIsFlattened(dltOutputSize)
tf = numel(dltOutputSize)==3 && isequal(dltOutputSize(1:2), [1 1]) || ...
    numel(dltOutputSize)==4 && isequal(dltOutputSize(1:3), [1 1 1]);
end
