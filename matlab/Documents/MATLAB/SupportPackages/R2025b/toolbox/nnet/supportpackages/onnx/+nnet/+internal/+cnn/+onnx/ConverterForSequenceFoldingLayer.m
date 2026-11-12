classdef ConverterForSequenceFoldingLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a nnet.cnn.layer.SequenceFoldingLayer into ONNX
    
    % Copyright 2019-2021 The MathWorks, Inc.
    
    methods
        function this = ConverterForSequenceFoldingLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            
            existingNodeNames   = {nodeProto.name};
            [onnxName, ~]       = legalizeNNTName(this, this.NNTLayer.Name);
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});
            isUndeterminedBatch = ~isnumeric(this.BatchSizeToExport);
            
            switch inputTensorLayout
                case 'snch'
                    shape        = [1 -1 0 0];
                    outputLayout = 'nch';
                case 'snchw'
                    shape        = [1 -1 0 0 0];
                    outputLayout = 'nchw';
                case 'snchwd'
                    shape        = [1 -1 0 0 0 0];
                    outputLayout = 'nchwd';
                otherwise
                    error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', this.NNTLayer.Name));
            end
            
            % Create a shape tensor, the second input to the
            % reshape node
            shapeTensor              = TensorProto;
            shapeTensor.name         = [onnxName '_Reshape_shape'];
            shapeTensor.data_type    = TensorProto_DataType.INT64;
            shapeTensor.raw_data     = rawData(int64(shape));
            shapeTensor.dims         = dimVector(numel(shape),1);
            
            % Create reshape node
            reshapeNodeName         = [onnxName '_Reshape'];
            reshapeNodeName         = makeUniqueName(existingNodeNames, reshapeNodeName);
            reshapeNode             = NodeProto;
            reshapeNode.op_type     = 'Reshape';
            reshapeNode.name        = reshapeNodeName;
            reshapeNode.input       = {inputTensorNames{1}, shapeTensor.name};
            reshapeNode.output      = {reshapeNodeName};
            newParams               = shapeTensor;
            
            % Create squeeze(0) node
            squeezeNodeName             = [onnxName '_Squeeze'];
            squeezeNodeName             = makeUniqueName([existingNodeNames, {reshapeNodeName}], squeezeNodeName);
            squeezeInput                = {reshapeNodeName};
            squeezeOutput               = {squeezeNodeName};
            [squeezeNode, squeezeInit]  = createNodeProto(this, 'Squeeze', squeezeNodeName, squeezeInput, squeezeOutput, 0);
            outputTensorName            = squeezeNodeName;
            newParams                   = [newParams, squeezeInit];
              
            if isUndeterminedBatch
                % if the batch size to export is undetermined, the fold
                % will also produce a Shape(X)[1] tensor indicating the
                % dynamic batch size. 
                sizeNodeName         = [onnxName '_InputSize'];
                sizeNodeName         = makeUniqueName([existingNodeNames, {reshapeNodeName, squeezeNodeName}], sizeNodeName); 
                sizeNode             = NodeProto;
                sizeNode.op_type     = 'Shape';
                sizeNode.name        = sizeNodeName;
                sizeNode.input       = inputTensorNames(1);
                sizeNode.output      = {sizeNodeName};
                
                % Constant tensor of value 1
                constantName         = [onnxName '_BatchIndex'];
                constantTensor           = TensorProto;
                constantTensor.name      = constantName;
                constantTensor.data_type = TensorProto_DataType.INT64;
                constantTensor.raw_data  = rawData(int64(1));
                constantTensor.dims      = dimVector(1,1); % 1-D tensor size 1. (Scalar) 
                
                % get Shape(X)[1] (runtime batch dimension)
                runtimeBatchName     = [onnxName '_RuntimeBatch']; 
                runtimeBatchName     = makeUniqueName([existingNodeNames, ...
                    {reshapeNodeName, squeezeNodeName, sizeNodeName}], runtimeBatchName);
                runtimeBatchNode     = NodeProto; 
                runtimeBatchNode.op_type = 'Gather'; 
                runtimeBatchNode.name    = runtimeBatchName; 
                runtimeBatchNode.input   = {sizeNodeName, constantName}; 
                runtimeBatchNode.output  = {runtimeBatchName};
                newParams                = [newParams constantTensor];  
            end
            
            if isUndeterminedBatch
                nodeProto               = [nodeProto sizeNode runtimeBatchNode reshapeNode squeezeNode];
            else 
                nodeProto               = [nodeProto reshapeNode squeezeNode];
            end 
            parameterInitializers   = newParams;
            networkInputs           = [];
            networkOutputs          = [];
            
            % Update maps
            origOutName                          = [this.NNTLayer.Name '/out'];
            TensorNameMap(origOutName)           = outputTensorName;
            TensorLayoutMap(outputTensorName)    = outputLayout;
            
            if isUndeterminedBatch
                minibatchOutputName                  = [this.NNTLayer.Name '/miniBatchSize']; 
                TensorNameMap(minibatchOutputName)   = runtimeBatchName; 
            end
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end
