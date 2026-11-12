classdef ConverterForClassificationOutputLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a ClassificationOutputLayer into ONNX
    
    % Copyright 2018-2022 The MathWorks, Inc.
    
    methods
        function this = ConverterForClassificationOutputLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            
            % Determine the network output tensor size
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});          
            
            % dltOutputSize can be [C], [1 C], [1 1 C], or [1 1 1 C] 
            dltOutputSize       = this.InputLayerSizes{1};                            
            
            switch inputTensorLayout
                case {'nchw', 'nchwd','nch'}
                    assert(DLTTensorIsFlattened(dltOutputSize),...
                        message('nnet_cnn_onnx:onnx:FlatInputRequired', this.NNTLayer.Name));                    
                    
                    % Insert a Flatten(1) operator before the existing
                    % softmax operator.
                    softmaxNodeName        = inputTensorNames{1};
                    flattenNodeName        = [softmaxNodeName '_Flatten'];
                    flattenNodeName        = makeUniqueName({nodeProto.name}, flattenNodeName);
                    FlattenNode            = NodeProto;
                    FlattenNode.op_type    = 'Flatten';
                    FlattenNode.name       = flattenNodeName;
                    FlattenNode.input      = {'dummyNode'};                % To be set later
                    FlattenNode.output     = {flattenNodeName};
                    FlattenNode.attribute  = makeAttributeProto('axis', 'INT', 1);                    
                    outputTensorName       = softmaxNodeName;
                    outputTensorSize       = [{this.BatchSizeToExport}, num2cell(dltOutputSize(end))];	% {N C} for CNNs.
                    nodeProto              = insertNodeBeforeSoftmax(this, nodeProto, FlattenNode, softmaxNodeName); 
                case 'snc'
                    assert(isscalar(dltOutputSize),...
                        message('nnet_cnn_onnx:onnx:FlatInputRequired', this.NNTLayer.Name));
                    
                    outputTensorName       = inputTensorNames{1};
                    outputTensorSize       = [{1 this.BatchSizeToExport}, num2cell(dltOutputSize)];	% {1 N C} for seq2seq RNNs.
                case 'nc'
                    assert(isscalar(dltOutputSize),...
                        message('nnet_cnn_onnx:onnx:FlatInputRequired', this.NNTLayer.Name));
                    
                    outputTensorName       = inputTensorNames{1};
                    outputTensorSize       = [{this.BatchSizeToExport}, num2cell(dltOutputSize)];
                case '1nc'
                    assert(isscalar(dltOutputSize),...
                        message('nnet_cnn_onnx:onnx:FlatInputRequired', this.NNTLayer.Name));
                    
                    % Insert a Flatten(2) operator before the existing
                    % softmax operator. Set the softmax axis to 1:
                    softmaxNodeName        = inputTensorNames{1};
                    flattenNodeName        = [softmaxNodeName '_Flatten'];
                    flattenNodeName        = makeUniqueName({nodeProto.name}, flattenNodeName);
                    FlattenNode            = NodeProto;
                    FlattenNode.op_type    = 'Flatten';
                    FlattenNode.name       = flattenNodeName;
                    FlattenNode.input      = {'dummyNode'};                            % To be set later
                    FlattenNode.output     = {flattenNodeName};
                    FlattenNode.attribute  = makeAttributeProto('axis', 'INT', 2);
                    outputTensorName       = softmaxNodeName;
                    outputTensorSize       = num2cell([this.BatchSizeToExport dltOutputSize]);     % {N C} for seq2last RNNs.
                    nodeProto              = insertNodeBeforeSoftmax(this, nodeProto, FlattenNode, softmaxNodeName, 1);
                case {'snchw', 'snchwd', 'snch'}
                    assert(DLTTensorIsFlattened(dltOutputSize),...
                        message('nnet_cnn_onnx:onnx:FlatInputRequired', this.NNTLayer.Name));
                    
                    softmaxNodeName          = inputTensorNames{1};                    
                    shape                    = [0 0 -1];
                    shapeTensor              = TensorProto;
                    shapeTensor.name         = [softmaxNodeName '_Reshape_shape'];
                    shapeTensor.name         = makeUniqueName({nodeProto.name}, shapeTensor.name);
                    shapeTensor.data_type    = TensorProto_DataType.INT64;
                    shapeTensor.raw_data     = rawData(int64(shape));
                    shapeTensor.dims         = dimVector(numel(shape),1);

                    reshapeNodeName        = [inputTensorNames{1} '_Reshape'];
                    reshapeNodeName        = makeUniqueName({nodeProto.name}, reshapeNodeName);
                    ReshapeNode            = NodeProto;
                    ReshapeNode.op_type    = 'Reshape';
                    ReshapeNode.name       = reshapeNodeName;
                    ReshapeNode.input      = {'dummyNode', shapeTensor.name};                          
                    ReshapeNode.output     = {reshapeNodeName};
                    outputTensorName       = softmaxNodeName;
                    outputTensorSize       = [{1 this.BatchSizeToExport}, num2cell(dltOutputSize(end))];     % {1 N C} for seq2last RNNs.
                    nodeProto              = insertNodeBeforeSoftmax(this, nodeProto, ReshapeNode, softmaxNodeName, 1);                   
                    
                otherwise
                    error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', this.NNTLayer.Name));
            end
            
            % Set outputs
            parameterInitializers   = [];
            networkInputs           = [];
            networkOutputs          = makeValueInfoProtoFromDimensions(outputTensorName, TensorProto_DataType.FLOAT, outputTensorSize);
        end
    end
end

function tf = DLTTensorIsFlattened(dltOutputSize)
tf = numel(dltOutputSize)==2 && isequal(dltOutputSize(1), 1) || ... 
    numel(dltOutputSize)==3 && isequal(dltOutputSize(1:2), [1 1]) || ...
    numel(dltOutputSize)==4 && isequal(dltOutputSize(1:3), [1 1 1]) ||...
    numel(dltOutputSize) == 1;
end

