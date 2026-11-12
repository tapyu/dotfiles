classdef ConverterForBatchNormalizationLayer < nnet.internal.cnn.onnx.exporter.NNTLayerConverter
    % Class to convert a batchNormalizationLayer into ONNX
    
    % Copyright 2018-2025 The MathWorks, Inc.
    
    methods
        function this = ConverterForBatchNormalizationLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.exporter.NNTLayerConverter(layerAnalyzer);
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
            %  BatchNorm. Move Sequence input from the first dim to the
            %  last dim
            [nodeProto, nextOperatorInput] = permuteSequenceDim(this, nodeProto, nextOperatorInput, TensorNameMap, inputTensorLayout, 1);

            % Make the nodeProto
            newNodeInput     = mapTensorNames(this,{nextOperatorInput, [onnxName '_scale'], [onnxName '_B'], [onnxName '_mean'], [onnxName '_var']}, TensorNameMap);
            [newNode, parameterInitializers] = createNodeProto(this, 'BatchNormalization', onnxName, ...
                newNodeInput, {onnxName}, this.NNTLayer.Epsilon,...
                this.NNTLayer.Scale, this.NNTLayer.Offset,...
                this.NNTLayer.TrainedMean, this.NNTLayer.TrainedVariance);
            
            nodeProto(end+1)        = newNode;
            networkInputs           = [];
            networkOutputs          = [];
            
            nextOperatorInput       = onnxName;
            %Output a sequence if the input was a sequence by permuting
            %the sequence dimension -> BCT to TBC
            [nodeProto, nextOperatorInput] = permuteSequenceDim(this, nodeProto, nextOperatorInput, TensorNameMap, inputTensorLayout, 0);

            % Update maps
            TensorNameMap(this.NNTLayer.Name) = nextOperatorInput;
            TensorLayoutMap(nextOperatorInput) = inputTensorLayout;
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end

