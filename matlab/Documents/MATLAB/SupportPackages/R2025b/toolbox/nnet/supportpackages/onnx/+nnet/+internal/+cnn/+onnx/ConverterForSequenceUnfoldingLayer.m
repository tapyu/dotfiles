classdef ConverterForSequenceUnfoldingLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a nnet.cnn.layer.SequenceUnfoldingLayer into ONNX
    
    % Copyright 2019-2021 The MathWorks, Inc.
    
    methods
        function this = ConverterForSequenceUnfoldingLayer(layerAnalyzer)
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
                case 'nch'
                    shape        = [0 0];
                    outputLayout = 'snch';
                case 'nchw'
                    shape        = [0 0 0];
                    outputLayout = 'snchw';
                case 'nchwd'
                    shape        = [0 0 0 0];
                    outputLayout = 'snchwd';
                otherwise
                    error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', this.NNTLayer.Name));
            end
            
            % Create unsqueeze(0) node
            unsqueezeNodeName                   = [onnxName '_Unsqueeze'];
            unsqueezeNodeName                   = makeUniqueName(existingNodeNames, unsqueezeNodeName);
            unsqueezeInput                      = {inputTensorNames{1}};
            unsqueezeOutput                     = {unsqueezeNodeName};
            [unsqueezeNode, unsqueezeInit]      = createNodeProto(this, 'Unsqueeze', unsqueezeNodeName, unsqueezeInput, unsqueezeOutput, 0);
            parameterInitializers               = unsqueezeInit;
            
            if isUndeterminedBatch 
                % Re-build reshape vector [-1, N, 0, 0, 0,...] to
                % dynamically re-shape the tensor like this: (SxN, C,...) ->
                % (S, N, C,...)
                seqDimName         = [onnxName '_SeqDim'];
                seqDimTensor         = TensorProto;
                seqDimTensor.name    = seqDimName;
                seqDimTensor.data_type = TensorProto_DataType.INT64;
                seqDimTensor.raw_data = rawData(int64(-1));
                seqDimTensor.dims    = dimVector(1,1); % 1-D tensor size 1

                featureDimName           = [onnxName '_FeatureDim'];
                featureDimTensor         = TensorProto;
                featureDimTensor.name    = featureDimName;
                featureDimTensor.data_type = TensorProto_DataType.INT64;
                featureDimTensor.raw_data = rawData(int64(shape));
                featureDimTensor.dims    = dimVector(numel(shape),1); % 1-D tensor size 3 or 4

                concatName = [onnxName '_Concat']; 
                concatName         = makeUniqueName([existingNodeNames, {unsqueezeNodeName}], concatName);
                concatNode         = NodeProto;
                concatNode.op_type = 'Concat'; 
                concatNode.name    = concatName; 
                concatNode.attribute = makeAttributeProto('axis', 'INT', 0);
                concatNode.input   = {seqDimName, inputTensorNames{2}, featureDimName};
                concatNode.output  = {concatName}; 
                shapeTensorName    = concatName; 
                newParams          = [seqDimTensor featureDimTensor]; 
            else 
                % Create a shape tensor, the second input to the
                % reshape node
                shapeTensor              = TensorProto;
                shapeTensor.name         = [onnxName '_Reshape_shape'];
                shapeTensor.data_type    = TensorProto_DataType.INT64;
                determinedShape          = [-1 this.BatchSizeToExport shape]; 
                shapeTensor.raw_data     = rawData(int64(determinedShape));
                shapeTensor.dims         = dimVector(numel(determinedShape),1);
                shapeTensorName          = shapeTensor.name; 
                newParams                = shapeTensor; 
            end 
            
            % Create reshape node
            reshapeNodeName         = [onnxName '_Reshape'];
            reshapeNodeName         = makeUniqueName([existingNodeNames, {unsqueezeNodeName}], reshapeNodeName);
            reshapeNode             = NodeProto;
            reshapeNode.op_type     = 'Reshape';
            reshapeNode.name        = reshapeNodeName;
            reshapeNode.input       = {unsqueezeNodeName, shapeTensorName};
            reshapeNode.output      = {reshapeNodeName};
            outputTensorName        = reshapeNodeName;
            
            if isUndeterminedBatch
                nodeProto = [nodeProto concatNode unsqueezeNode reshapeNode];  
            else 
                nodeProto = [nodeProto unsqueezeNode reshapeNode];
            end
            
            parameterInitializers   = [parameterInitializers newParams];
            networkInputs           = [];
            networkOutputs          = [];
            
            % Update maps
            TensorNameMap(this.NNTLayer.Name) = outputTensorName;
            TensorLayoutMap(outputTensorName) = outputLayout;
            
            
           % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end
