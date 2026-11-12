classdef ConverterForGlobalAveragePooling2dLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a builtin globalAveragePooling2DLayer into ONNX
        
    % Copyright 2019-2022 The MathWorks, Inc.

    methods
        function this = ConverterForGlobalAveragePooling2dLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*

            nextOperatorInput = this.InputLayerNames{1};
            
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});
            
            if isequal(inputTensorLayout,'snch')
                 [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [1 2 3 0]);
            elseif isequal(inputTensorLayout,'snchw')
                 [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, nextOperatorInput, TensorNameMap, [1 2 3 4 0]);
            end

            if ismember(inputTensorLayout,{'snch','nchw'})
                
                % GAP node
                [gapName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
                gapName                = makeUniqueName({nodeProto.name}, gapName);
                gapNode              = NodeProto;
                gapNode.op_type      = 'GlobalAveragePool';
                gapNode.name         = gapName;
                gapNode.input        = mapTensorNames(this, {nextOperatorInput}, TensorNameMap);
                gapNode.output       = {gapName};
                
                nodeProto(end+1)        = gapNode;
                parameterInitializers   = [];
                networkInputs           = [];
                networkOutputs          = [];

                nextOperatorInput       = gapName;
                outputTensorLayout      = 'nchw';
                if isequal(inputTensorLayout,'snch')
                    %Squeeze nch1 to nch
                    squeezeName                 = [gapName '_Squeeze'];
                    squeezeName                 = makeUniqueName([{nodeProto.name}, {nextOperatorInput}], squeezeName);
                    squeezeInput                = {gapName};
                    squeezeOutput               = {squeezeName};
                    [squeezeNode, squeezeInit]  = createNodeProto(this, 'Squeeze', squeezeName, squeezeInput, squeezeOutput, 3);                    
                    nodeProto(end+1)            = squeezeNode;
                    parameterInitializers       = [parameterInitializers squeezeInit];
                    
                    nextOperatorInput      =  squeezeName;
                    outputTensorLayout     = 'nch';
                end
                % Update maps
                TensorNameMap(this.NNTLayer.Name) = nextOperatorInput;
                TensorLayoutMap(nextOperatorInput)= outputTensorLayout;
            elseif isequal(inputTensorLayout,'snchw')
                % GAP node
                [gapName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
                gapName                = makeUniqueName({nodeProto.name}, gapName);
                gapNode                = NodeProto;
                gapNode.op_type        = 'AveragePool';
                gapNode.name           = gapName;
                gapNode.input          = mapTensorNames(this, {nextOperatorInput}, TensorNameMap);
                gapNode.output         = {gapName};
                gapNode.attribute      = makeAttributeProto('kernel_shape','INTS',[this.InputLayerSizes{1}(1:2) 1]);
                
                nodeProto(end+1)        = gapNode;
                parameterInitializers   = [];
                networkInputs           = [];
                networkOutputs          = [];
                
                [nodeProto, nextOperatorInput] = appendTransposeOperator(this, nodeProto, gapName, TensorNameMap, [4 0 1 2 3]);
                
                % Update maps
                TensorNameMap(this.NNTLayer.Name)           = nextOperatorInput;
                TensorLayoutMap(nextOperatorInput)          = 'snchw';
            end
            

            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end
