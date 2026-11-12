classdef ConverterForONNXCustomOutputLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a CustomOutputLayer into ONNX
    
    % Copyright 2024 The MathWorks, Inc.
    
    methods
        function this = ConverterForONNXCustomOutputLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            % Collect information from current node. Determine the network output
            % tensor size.
            inputTensorNames   = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            inputTensorLayout       = TensorLayoutMap(inputTensorNames{1});
            dltOutputSize            = this.OutputSize{1};
            
            newNode = [];
            
            switch inputTensorLayout
                case {'nch', 'nchw', 'nchwd'}
                    if DLTTensorIsFlattened(dltOutputSize)
                        % If dltOutputSize is flattened, add a Flatten(1) operator
                        flattenNodeName        = [inputTensorNames{1} '_Flatten'];
                        flattenNodeName        = makeUniqueName({nodeProto.name}, flattenNodeName);
                        FlattenNode            = NodeProto;
                        FlattenNode.op_type    = 'Flatten';
                        FlattenNode.name       = flattenNodeName;
                        FlattenNode.output     = {flattenNodeName};
                        FlattenNode.attribute  = makeAttributeProto('axis', 'INT', 1);
                        
                        % Output a Flatten(1) operator
                        FlattenNode.input      = inputTensorNames(1);
                        outputTensorName       = flattenNodeName;
                        newNode                = FlattenNode;
                        
                        outputTensorSize       = [{this.BatchSizeToExport}, num2cell(dltOutputSize(end))];	% {N C}
                    else
                        % If dltOutputSize is not flattened, create output tensor
                        % directly
                        outputTensorName       = inputTensorNames{1};
                        outputTensorSize       = [{this.BatchSizeToExport}, num2cell(dltOutputSize([end 1:(end-1)]))];	% {N C H W} or {N C H W D}
                    end
                    
                case 'snc'
                    if ~isscalar(dltOutputSize)
                        error(message('nnet_cnn_onnx:onnx:FlatOutputRequired', this.NNTLayer.Name));
                    end
                    
                    outputTensorName       = inputTensorNames{1};
                    outputTensorSize       = [{1 this.BatchSizeToExport}, num2cell(dltOutputSize)];	% {1 N C}
                case 'nc'
                    if ~isscalar(dltOutputSize)
                        error(message('nnet_cnn_onnx:onnx:FlatOutputRequired', this.NNTLayer.Name));
                    end
                    
                    outputTensorName       = inputTensorNames{1};
                    outputTensorSize       = [{this.BatchSizeToExport}, num2cell(dltOutputSize)];	% {N C}
                case '1nc'
                    if ~isscalar(dltOutputSize)
                        error(message('nnet_cnn_onnx:onnx:FlatOutputRequired', this.NNTLayer.Name));
                    end

                    flattenNodeName        = [inputTensorNames{1} '_Flatten'];
                    flattenNodeName        = makeUniqueName({nodeProto.name}, flattenNodeName);
                    FlattenNode            = NodeProto;
                    FlattenNode.op_type    = 'Flatten';
                    FlattenNode.name       = flattenNodeName;
                    FlattenNode.output     = {flattenNodeName};
                    FlattenNode.attribute  = makeAttributeProto('axis', 'INT', 2);

                    % Output a Flatten(2) operator
                    FlattenNode.input      = inputTensorNames(1);
                    outputTensorName       = flattenNodeName;
                    newNode                = FlattenNode;
               
                    outputTensorSize       = [{this.BatchSizeToExport}, num2cell(dltOutputSize)];	% {N C}
                    
                case {'snch','snchw', 'snchwd'}
                    % Create output tensor directly
                    outputTensorName        = inputTensorNames{1};
                    outputTensorSize        = [{1 this.BatchSizeToExport}, num2cell(dltOutputSize([end 1:(end-1)]))];	%{1 N C H}, {1 N C H W}, or {1 N C H W D}
                    
                otherwise
                    error(message('nnet_cnn_onnx:onnx:UnexpectedONNXOutputLayout', this.NNTLayer.Name));
            end
            
            % Set outputs
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
