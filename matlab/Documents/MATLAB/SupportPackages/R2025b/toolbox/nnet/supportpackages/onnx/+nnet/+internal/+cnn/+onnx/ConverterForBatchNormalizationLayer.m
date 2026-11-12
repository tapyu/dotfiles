classdef ConverterForBatchNormalizationLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a batchNormalizationLayer into ONNX
    
    % Copyright 2018-2022 The MathWorks, Inc.
    
    methods
        function this = ConverterForBatchNormalizationLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            onnxName                = makeUniqueName({nodeProto.name}, onnxName);
            inputTensorNames            = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout           = TensorLayoutMap(inputTensorNames{1});
            
            
             nextOperatorInput = this.InputLayerNames{1};
            
            %  Permute Sequence input dimension to Spatial Dimension for
            
            %  BatchNorm to operate over TBC -> BCT
            if isequal(inputTensorLayout,'snc')
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [1 2 0]);
                %  BatchNorm to operate over TBCS -> BCST
            elseif isequal(inputTensorLayout,'snch')
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [1 2 3 0]);
                %  BatchNorm to operate over TBCSS -> BCSST
            elseif isequal(inputTensorLayout,'snchw')
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [1 2 3 4 0]);
                %  BatchNorm to operate over TBCSSS -> BCSSST
            elseif isequal(inputTensorLayout,'snchwd')
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [1 2 3 4 5 0]);
            end

            % Make the nodeProto
            newNode           = NodeProto;
            newNode.op_type	= 'BatchNormalization';
            newNode.name      = onnxName;
            newNode.input     = mapTensorNames(this,{nextOperatorInput, [onnxName '_scale'], [onnxName '_B'], [onnxName '_mean'], [onnxName '_var']}, TensorNameMap);
            newNode.output    = {onnxName};
            newNode.attribute = makeAttributeProto('epsilon', 'FLOAT', this.NNTLayer.Epsilon);
            if this.OpsetVersion < 7
                newNode.attribute(1,2) = makeAttributeProto('is_test', 'INT', 1);
            end
            
            % Make parameter Initializers for: scale, B, mean, var
            t1 = TensorProto;
            t1.name = [onnxName '_scale'];
            t1.data_type = TensorProto_DataType.FLOAT;
            t1.raw_data = rawData(single(this.NNTLayer.Scale));
            t1.dims = dimVector(numel(this.NNTLayer.Scale),1);           % NNT data: numChannels -1 (1D input) or 1-1-numChannels (2D input) or 1-1-1-numChannels (3D input)
            
            t2 = TensorProto;
            t2.name = [onnxName '_B'];
            t2.data_type = TensorProto_DataType.FLOAT;
            t2.raw_data = rawData(single(this.NNTLayer.Offset));
            t2.dims = dimVector(numel(this.NNTLayer.Offset),1);          % NNT data: numChannels -1 (1D input) or 1-1-numChannels (2D input) or 1-1-1-numChannels (3D input)
            
            t3 = TensorProto;
            t3.name = [onnxName '_mean'];
            t3.data_type = TensorProto_DataType.FLOAT;
            t3.raw_data = rawData(single(this.NNTLayer.TrainedMean));
            t3.dims = dimVector(numel(this.NNTLayer.TrainedMean),1);  	% NNT data: numChannels -1 (1D input) or 1-1-numChannels (2D input) or 1-1-1-numChannels (3D input)
            
            t4 = TensorProto;
            t4.name = [onnxName '_var'];
            t4.data_type = TensorProto_DataType.FLOAT;
            t4.raw_data = rawData(single(this.NNTLayer.TrainedVariance));
            t4.dims = dimVector(numel(this.NNTLayer.TrainedVariance),1);	% NNT data: numChannels -1 (1D input) or 1-1-numChannels (2D input) or 1-1-1-numChannels (3D input)
            
            parameterInitializers = [t1 t2 t3 t4];
            
            nodeProto(end+1)        = newNode;
            networkInputs           = [];
            networkOutputs          = [];
            
            nextOperatorInput       = onnxName;
            %Output a sequence if the input was a sequence by permuting
            %the sequence dimension -> BCT to TBC
            if isequal(inputTensorLayout,'snc')
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [2 0 1]);
                %the sequence dimension -> BCST to TBCS
            elseif isequal(inputTensorLayout,'snch')
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [3 0 1 2]);
                %the sequence dimension -> BCSST to TBCSS
            elseif isequal(inputTensorLayout,'snchw')
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [4 0 1 2 3]);
                %the sequence dimension -> BCSSST to TBCSSS
            elseif isequal(inputTensorLayout,'snchwd')
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [5 0 1 2 3 4]);
            end

            % Update maps
            TensorNameMap(this.NNTLayer.Name) = nextOperatorInput;
            TensorLayoutMap(nextOperatorInput) = inputTensorLayout;
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end

